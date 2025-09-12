# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide
from megatron.training.global_vars import get_args
from megatron.training.training import unwrap_model
from megatron.training.utils import average_losses_across_data_parallel_group
from omegaconf import DictConfig
from torch.multiprocessing.reductions import reduce_tensor

from rlinf.algorithms.math.algo_functions import (
    actor_loss_fn,
    calculate_adv_and_returns,
    kl_penalty,
)
from rlinf.algorithms.math.verifier.verify import math_verify_call
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.megatron.megatron_model_manager import (
    MegatronModelManager,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_dynamic,
    get_iterator_k_split,
    get_last_rank,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    broadcast_tensor_within_mp,
    broadcast_tensor_within_pp,
    compute_rollout_metrics,
    compute_rollout_metrics_pipeline,
    masked_normalization,
    vocab_parallel_entropy_and_log_probs,
    vocab_parallel_log_probs_from_logits,
)
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.profiler import PyTorchProfiler, PyTorchProfilerFunc
from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
from rlinf.utils.resharding.reshard_config import ReshardConfig
from rlinf.utils.train_utils import (
    set_eval,
    set_sync_funcs,
    set_train,
)
from rlinf.utils.utils import (
    clear_memory,
    configure_batch_sizes,
    cpu_dict,
    cpu_weight_swap,
    masked_mean,
    retrieve_model_state_dict_in_cpu,
    seq_mean_token_mean,
    seq_mean_token_sum,
)
from rlinf.workers.rollout.utils import (
    DisaggRankMapper,
    HybridRankMapper,
)

try:
    from params_resharding import resharding_init
    HAVE_RESHARDING = True
except ImportError:
    print("can't find params_resharding, resharding is not supported", flush=True)
    HAVE_RESHARDING = False

class MegatronActor(MegatronModelManager, Worker):
    """The class for running the actor training using Megatron."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role="actor"
    ):
        """Initialize the MegatronActor.

        Args:
            cfg (DictConfig): The configuration for the actor.
        """
        Worker.__init__(self)
        role_cfg = getattr(cfg, role, None)
        if role_cfg is None:
            raise ValueError(f"Role {role} is not defined in the configuration.")
        super().__init__(role_cfg)
        self.cfg = cfg

        self.response_len = (
            role_cfg.model.encoder_seq_length - cfg.data.max_prompt_length
        )

        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.ratio_eps = self.cfg.algorithm.ratio_clip_eps
        self.logprob_forward_micro_batch_size = (
            self.cfg.algorithm.logprob_forward_micro_batch_size
        )

        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type
        self.clip_ratio_c = self.cfg.algorithm.clip_ratio_c

        if self.cfg.algorithm.loss_agg_func == "token-mean":
            self.loss_agg_func = masked_mean
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-sum":
            self.loss_agg_func = seq_mean_token_sum
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-mean":
            self.loss_agg_func = seq_mean_token_mean
        else:
            raise NotImplementedError(
                f"algorithm.loss_agg_func={self.cfg.algorithm.loss_agg_func} is not supported!"
            )

        self.enable_dynamic_batch_size = self.cfg.runner.enable_dynamic_batch_size
        self.max_tokens_per_mbs = self.cfg.runner.max_tokens_per_mbs

        self.ref_policy_state_dict = None

        self.offload_optimizer = self.cfg.actor.offload_optimizer
        self.offload_weight = self.cfg.actor.offload_weight
        self.offload_grad = self.cfg.actor.offload_grad

        if not self.cfg.reward.use_reward_model:
            assert self.cfg.reward.reward_type == "math", "only support math"
            self.reward_fn = math_verify_call

        self._rollout_group_name = self.cfg.rollout.group_name

        self._is_data_io_rank = (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_context_parallel_rank() == 0
            and parallel_state.get_pipeline_model_parallel_rank() == 0
        )

        self.component_placement = placement

        if self.component_placement.placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]:
            self._data_channel = self.connect_channel(
                channel_name=role_cfg.channel.name
            )
            self._queue_name = f"{role_cfg.channel.queue_name}"
            self._data_channel.create_queue(
                self._queue_name, maxsize=role_cfg.channel.queue_size
            )

            self.batch_iterator = BatchResizingIterator(
                fetch_batch_fn=self.get_batch_fn,
                micro_batch_size=role_cfg.micro_batch_size,
                global_batch_size_per_dp=role_cfg.global_batch_size
                // parallel_state.get_data_parallel_world_size(),
            )

            self._batch_buffer_for_metrics: List[RolloutResult] = []

        self.average_respone_len = self.response_len

        self._init_profiler()

        # TODO(Chunyang && Junhao) :: megatron + scheduler
        try:
            self.use_schedule = self.cfg.cluster.placement.use_schedule
        except Exception as e:
            self.use_schedule = False
        
        self.use_auto_scheduler = (self.component_placement._placement_mode == PlacementMode.AUTO)
        self.use_pre_process_policy = self.cfg.cluster.use_pre_process_policy and self.use_auto_scheduler
        self.resharding_state = True
        if self.use_auto_scheduler:
            from rlinf.scheduler.dynamic_scheduler.util import (
                get_scheduler_channel,
                get_scheduler_request_queue,
                get_scheduler_response_queue,
            )
            assert HAVE_RESHARDING, "params_resharding is required for scheduler"
            self.schedule_channel = self.connect_channel(get_scheduler_channel(role))
            # warmup
            self.schedule_channel.put(None, async_op=False)
            self.schedule_channel.get(async_op=False)
            
            self.scheduler_request_queue = get_scheduler_request_queue(self._rank)
            self.scheduler_response_queue = get_scheduler_response_queue(self._rank)

    def _init_profiler(self):
        def _validate_schedule_info():
            assert (
                self.cfg.actor.megatron.profiler.schedule_warmup is not None
                and self.cfg.actor.megatron.profiler.schedule_warmup >= 0
            ), "<schedule_warmup> must be set and greater than 0 when using profiler."
            assert (
                self.cfg.actor.megatron.profiler.schedule_active is not None
                and self.cfg.actor.megatron.profiler.schedule_active > 0
            ), "<schedule_active> must be set and greater than 0 when using profiler."

        self.use_profiler = self.cfg.actor.megatron.use_profiler

        # here we should validate profiler's schedule info
        if self.use_profiler:
            _validate_schedule_info()
        self.profiler = (
            PyTorchProfiler.from_config(self.cfg.actor.megatron.profiler)
            if self.use_profiler
            else None
        )
        self._forward_only_record = PyTorchProfilerFunc(
            "forward_only", self.use_profiler
        )
        self._dynamic_batch_processing_record = PyTorchProfilerFunc(
            "dynamic_batch_processing", self.use_profiler
        )
        self._static_batch_processing_record = PyTorchProfilerFunc(
            "static_batch_processing", self.use_profiler
        )
        self._broadcast_outputs_record = PyTorchProfilerFunc(
            "broadcast_outputs", self.use_profiler
        )

        self._megatron_forward_backward_record = PyTorchProfilerFunc(
            "megatron_forward_backward", self.use_profiler
        )

    def init_worker(self):
        self.setup_model_and_optimizer()
        
        if self.use_auto_scheduler:
            self.init_trainer_resharding()
            if not self.resharding_state:
                return

        ref_policy_state_dict = None
        # only need this if we are running with inital kl penalty & full-parameter tuning
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model[0])
        self.ref_policy_state_dict = ref_policy_state_dict

        rollout_reshard_config = ReshardConfig(
            model_arch=self.cfg.rollout.model_arch,
            model_config=self.transformer_config,
            reshard_tp_size=self.cfg.rollout.tensor_parallel_size,
            reshard_pp_size=self.cfg.rollout.pipeline_parallel_size,
        )
        self.rollout_weights_reshard = MegatronCoreWeightReshard(rollout_reshard_config)
        self._setup_rollout_weight_dst_ranks()

        if self.cfg.get("inference", None) is not None:
            inference_reshard_config = ReshardConfig(
                model_arch=self.cfg.inference.model_arch,
                model_config=self.transformer_config,
                reshard_weights_format="mcore",
                reshard_tp_size=self.cfg.inference.model.tensor_model_parallel_size,
                reshard_pp_size=self.cfg.inference.model.pipeline_model_parallel_size,
            )
            self.inference_weights_reshard = MegatronCoreWeightReshard(
                inference_reshard_config
            )
            self._setup_inference_weight_dst_ranks()

        # Create GLOO MP group for broadcast
        self._mp_group_ranks = parallel_state._MODEL_PARALLEL_GLOBAL_RANKS

        torch.distributed.barrier()

    def get_forward_step_func(self):
        """Acquire the forward step function for the model."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            batch = {key: val.cuda() for key, val in batch.items()}

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            response_len = self.response_len
            responses = input_ids[:, -response_len:]
            label = copy.deepcopy(position_ids)
            label[:, -response_len - 1 : -1] = responses
            label_mask = copy.deepcopy(attention_mask)
            label_mask[:, : -response_len - 1] = False
            label_mask[:, -1] = False

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape

                if self.calculate_entropy:
                    entropy, log_probs = vocab_parallel_entropy_and_log_probs(
                        logits,
                        label,
                        calculate_entropy_loss=self.calculate_entropy_loss,
                    )
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret = {"log_probs": log_probs, "entropy": entropy}
                else:
                    log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret = {"log_probs": log_probs}

                return ret

            logits_processor_args = {"label": label, "label_mask": label_mask}

            output = self.custom_forward(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.transformer_config.sequence_parallel,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                temperature=self.cfg.algorithm.sampling_params.temperature,
            )

            if not self.return_loss:

                def id_func(output, non_loss_data=True):
                    return output["log_probs"][:, -response_len - 1 : -1].contiguous()

                return output, id_func
            else:

                def loss_func(output):
                    curr_logprobs = output["log_probs"][
                        :, -response_len - 1 : -1
                    ].contiguous()

                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    ref_logprobs = None
                    if "ref_logprobs" in batch:
                        ref_logprobs = batch["ref_logprobs"]

                    mask = batch["attention_mask"][:, -response_len:]

                    # Calculate clipped PPO surrogate loss function.
                    (
                        loss,
                        proportion_clipped,
                        approx_kl,
                        ratios,
                        cliped_ratio,
                        dual_cliped_ratio,
                    ) = actor_loss_fn(
                        self.loss_agg_func,
                        curr_logprobs,
                        prev_logprobs,
                        advantages,
                        self.ratio_eps,
                        mask,
                    )

                    logging_loss = loss.detach()
                    entropy_loss = torch.zeros(1, device=loss.device)
                    if self.calculate_entropy:
                        entropy = output["entropy"][
                            :, -response_len - 1 : -1
                        ].contiguous()
                        entropy_loss = self.loss_agg_func(entropy, mask=mask)
                        if self.calculate_entropy_loss:
                            loss = (
                                loss - self.cfg.algorithm.entropy_bonus * entropy_loss
                            )

                    kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if self.kl_beta > 0 and ref_logprobs is not None:
                        kld = kl_penalty(
                            ref_logprobs, curr_logprobs, self.kl_penalty_type
                        )
                        kl_loss = self.loss_agg_func(kld, mask)
                        loss = loss + kl_loss * self.kl_beta

                    # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
                    _imp = (ratios.detach().float() * mask).sum()
                    torch.distributed.all_reduce(
                        _imp, group=parallel_state.get_data_parallel_group()
                    )
                    _n_valid_tokens = mask.count_nonzero().clone()
                    torch.distributed.all_reduce(
                        _n_valid_tokens, group=parallel_state.get_data_parallel_group()
                    )
                    _imp /= _n_valid_tokens
                    # Early stopping.
                    if (
                        self.cfg.algorithm.early_stop_imp_ratio is not None
                        and _imp > self.cfg.algorithm.early_stop_imp_ratio
                    ):
                        self.log_warning(
                            f"Current importance ratio {_imp.item():.4f} is larger "
                            f"than early stop threshold {self.cfg.algorithm.early_stop_imp_ratio}. Abandon this microbatch."
                        )
                        loss = loss * 0.0
                    if self.cfg.algorithm.use_valid_token_scale:
                        loss_scale = (
                            mask.sum()
                            / self.global_valid_token
                            * parallel_state.get_data_parallel_world_size()
                            * self.num_microbatches
                        )
                        loss *= loss_scale.item()

                    with torch.no_grad():
                        ratios = masked_mean(ratios.detach(), mask)
                        cliped_ratio = masked_mean(cliped_ratio.detach(), mask)
                        dual_cliped_ratio = masked_mean(
                            dual_cliped_ratio.detach(), mask
                        )
                        entropy_loss = entropy_loss.detach()
                        kl_loss = kl_loss.detach()
                        approx_kl = approx_kl.detach()
                        proportion_clipped = proportion_clipped.detach()

                    (
                        reduced_actor_loss,
                        ratios,
                        cliped_ratio,
                        dual_cliped_ratio,
                        entropy_loss,
                        kl_loss,
                        approx_kl,
                        proportion_clipped,
                    ) = average_losses_across_data_parallel_group(
                        [
                            logging_loss,
                            ratios,
                            cliped_ratio,
                            dual_cliped_ratio,
                            entropy_loss,
                            kl_loss,
                            approx_kl,
                            proportion_clipped,
                        ]
                    )
                    return (
                        loss,
                        {
                            "loss": reduced_actor_loss,
                            "ratio": ratios,
                            "cliped_ratio": cliped_ratio,
                            "dual_cliped_ratio": dual_cliped_ratio,
                            "entropy_loss": entropy_loss,
                            "kl_loss": kl_loss,
                            "approx_kl": approx_kl,
                            "proportion_clipped": proportion_clipped,
                        },
                    )

                return output, loss_func

        return forward_output_and_loss_func

    def run_forward_backward(self, batch, forward_only=True):
        """Run the forward and backward pass on the model.

        Args:
            batch (dict): The input batch containing the data for the forward pass.
            forward_only (bool): If True, only run the forward pass without backpropagation.
        """
        clear_memory()
        batch_size = batch["input_ids"].size(0)
        sequence_length = batch["input_ids"].size(1)

        if self.enable_dynamic_batch_size:
            max_tokens_per_mbs = (
                self.max_tokens_per_mbs
                * parallel_state.get_context_parallel_world_size()
            )
            vpp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = (
                    self.transformer_config.microbatch_group_size_per_vp_stage
                )
                data_iter, indices, n_micro_batch = get_iterator_dynamic(
                    batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_tokens_per_mbs=max_tokens_per_mbs,
                )
                assert (
                    n_micro_batch
                    % self.transformer_config.microbatch_group_size_per_vp_stage
                    == 0
                ), (
                    f"micro_batches {data_iter} must be divisible by microbatch_group_size_per_vp_stage {microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                data_iter, indices, n_micro_batch = get_iterator_dynamic(
                    batch, max_tokens_per_mbs=max_tokens_per_mbs
                )
            total_seqlen = max_tokens_per_mbs
        else:
            forward_micro_batch_size = (
                self.logprob_forward_micro_batch_size
                if forward_only
                else self.cfg.actor.micro_batch_size
            )
            n_micro_batch = (
                max(1, batch_size // forward_micro_batch_size)
                if forward_only
                else get_num_microbatches()
            )
            data_iter = get_iterator_k_split(batch, n_micro_batch)
            total_seqlen = forward_micro_batch_size * sequence_length
        fwd_bwd_function = get_forward_backward_func()

        self.num_microbatches = n_micro_batch

        if self.use_profiler:
            self.profiler.start(forward_only=forward_only)

        self.return_loss = not forward_only
        self._forward_only_record.start()
        forward_outputs = fwd_bwd_function(
            forward_step_func=self.get_forward_step_func(),
            data_iterator=self.make_data_iterator_list(data_iter, padding=True),
            model=self.model,
            num_microbatches=n_micro_batch,
            forward_only=forward_only,
            seq_length=total_seqlen,
            micro_batch_size=1,
            collect_non_loss_data=True if forward_only else False,
        )
        self._forward_only_record.stop()

        if forward_only:
            if self.enable_dynamic_batch_size:
                self._dynamic_batch_processing_record.start()
                outputs = torch.cat(forward_outputs, dim=0).to(torch.float32)
                indices = sum(indices, [])
                assert len(indices) == outputs.size(0), (
                    f"{len(indices)} vs. {outputs.size()}"
                )
                revert_indices = torch.tensor(
                    get_reverse_idx(indices), dtype=torch.long
                )
                outputs = outputs[revert_indices]
                self._dynamic_batch_processing_record.stop()
            else:
                self._static_batch_processing_record.start()
                outputs = (
                    torch.cat(forward_outputs) if len(forward_outputs) > 0 else None
                )
                self._static_batch_processing_record.stop()

            self._broadcast_outputs_record.start()
            outputs = broadcast_tensor_within_pp(outputs)
            self._broadcast_outputs_record.stop()
        else:
            outputs = {}

            if forward_outputs:
                keys = forward_outputs[0].keys()
                for key in keys:
                    metric_mean = torch.stack(
                        [loss_reduced[key] for loss_reduced in forward_outputs]
                    ).mean()
                    torch.distributed.broadcast(metric_mean, get_last_rank())

                    outputs[key] = metric_mean.cpu().item()

        if self.use_profiler:
            self.profiler.stop(forward_only=forward_only)

        return outputs

    def run_forward_backward_pipeline(self, batch, forward_only=False):
        sequence_length = self.cfg.runner.seq_length
        fwd_bwd_function = get_forward_backward_func()
        forward_micro_batch_size = self.cfg.actor.micro_batch_size
        n_micro_batch = get_num_microbatches()
        self.num_microbatches = n_micro_batch
        self.return_loss = True
        forward_outputs = fwd_bwd_function(
            forward_step_func=self.get_forward_step_func(),
            data_iterator=self.make_data_iterator_list(batch),
            model=self.model,
            num_microbatches=n_micro_batch,
            forward_only=False,
            seq_length=forward_micro_batch_size * sequence_length,
            micro_batch_size=1,
        )
        outputs = {}

        for key in [
            "loss",
            "ratio",
            "cliped_ratio",
            "dual_cliped_ratio",
            "entropy_loss",
            "kl_loss",
            "approx_kl",
            "proportion_clipped",
        ]:
            if forward_outputs:
                metric_mean = torch.stack(
                    [loss_reduced[key] for loss_reduced in forward_outputs]
                ).mean()
            else:
                metric_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            torch.distributed.broadcast(metric_mean, get_last_rank())

            outputs[key] = metric_mean.cpu().item()

        return outputs

    # Training
    def get_batch_fn(self):
        if (
            parallel_state.get_pipeline_model_parallel_rank() == 0
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            result: RolloutResult = self._data_channel.get(queue_name=self._queue_name)
            torch.distributed.broadcast_object_list(
                [result],
                device=torch.cuda.current_device(),
                src=parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            self._batch_buffer_for_metrics.append(result)

            batch = result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.runner.seq_length,
                pad_token=self.tokenizer.eos_token_id,
            )
            batch["prev_logprobs"] = result.prev_logprobs
            batch["ref_logprobs"] = result.ref_logprobs
            return batch
        else:
            results: List[RolloutResult] = [None]
            torch.distributed.broadcast_object_list(
                results,
                device=torch.cuda.current_device(),
                src=parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

            self._batch_buffer_for_metrics.append(results[0])
            batch = results[0].to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.runner.seq_length,
                pad_token=self.tokenizer.eos_token_id,
            )
            batch["prev_logprobs"] = results[0].prev_logprobs
            batch["ref_logprobs"] = results[0].ref_logprobs

            return batch

    def training_step(self, batch, pipeline_func: bool = False):
        """Run a single training step on the model.

        Args:
            batch (dict): The input batch containing the data for the forward pass.
        """
        set_sync_funcs(self, forward_only=False)
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        if self.cfg.algorithm.use_valid_token_scale:
            if pipeline_func:
                self.global_valid_token = (
                    self.average_respone_len
                    * get_num_microbatches()
                    * self.cfg.actor.micro_batch_size
                )
            else:
                loss_mask = batch["attention_mask"][:, -self.response_len :]
                global_valid_token = loss_mask.to(dtype=torch.float32).sum().cuda()
                torch.distributed.all_reduce(
                    global_valid_token, group=parallel_state.get_data_parallel_group()
                )
                self.global_valid_token = global_valid_token

        if pipeline_func:
            metrics = self.run_forward_backward_pipeline(batch, forward_only=False)
        else:
            metrics = self.run_forward_backward(batch, forward_only=False)
        increment = (
            get_num_microbatches()
            * self.cfg.actor.micro_batch_size
            * parallel_state.get_data_parallel_world_size()
        )
        success, grad_norm, num_zeros_in_grad, lr = self.optimizer_step(increment)

        metrics["grad_norm"] = grad_norm if grad_norm is not None else float("nan")
        metrics["num_zeros_in_grad"] = (
            num_zeros_in_grad if num_zeros_in_grad is not None else float("nan")
        )
        metrics["lr"] = lr if lr is not None else float("nan")
        metrics["update_success"] = int(success)

        return metrics

    def run_training(self):
        """Run the training loop for the actor."""
        set_train(self)
        configure_batch_sizes(
            rank=torch.distributed.get_rank(),
            mbs=self.cfg.actor.micro_batch_size,
            gbs=self.cfg.actor.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
        rollout_size = self.rollout_batches["input_ids"].size(0)
        num_microbatches = divide(
            rollout_size,
            self.cfg.actor.global_batch_size
            // parallel_state.get_data_parallel_world_size(),
        )
        rollout_dataloader_iter = get_iterator_k_split(
            batch=self.rollout_batches,
            num_microbatches=num_microbatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )
        training_metrics_list = []

        if self.use_profiler:
            self.profiler.init_fwd_bwd_schedule(num_microbatches)
        for batch in rollout_dataloader_iter:
            training_metrics = self.training_step(batch)
            training_metrics_list.append(training_metrics)

        return training_metrics_list

    def sync_with_scheduler(self):
        if not self.use_auto_scheduler:
            return

        if self._rank == 0:
            self.log_info(f"[dev-hjh] Actor update state to scheduler, using queue {self.scheduler_response_queue}")
            self.schedule_channel.put(
                None, queue_name=self.scheduler_response_queue, async_op=True
            ).wait()

            self.log_info(f"[dev-hjh] Actor start recv resharding_resp, using queue {self.scheduler_request_queue}")
            resharding_resp = self.schedule_channel.get(
                queue_name=self.scheduler_request_queue,
                async_op=True,
            ).wait()

            self.log_info(f"[dev-hjh] Actor finish recv resharding_resp, resharding_resp={resharding_resp}")
        else:
            resharding_resp = None
        
        parallel_state.barrier_with_gloo()
        resharding_resp = self.broadcast_obj(resharding_resp)
        self.log_info(f"[debug-hjh] rank={self._rank}, resharding_resp={resharding_resp}")
        
        if resharding_resp is not None:
            self.apply_parallel_strategy(resharding_resp)
            self.calc_num_microbatches()
       
        


    def run_training_pipeline(self):
        # dev0910
        if self.use_pre_process_policy:
            self.log_info(f"[dev-hjh] Actor wait for main_loop")
            if self._rank == 0:
                self.schedule_channel.get(queue_name=self.scheduler_request_queue,async_op=True).wait()
            parallel_state.barrier_with_gloo()
            self.log_info(f"[dev-hjh] Actor finish for main_loop")
            if self.resharding_state:
                self.onload_model_weights_and_grad(load_grad=self.offload_grad)
                self.onload_megatron_optimizer()
        
        set_train(self)
        self.calc_num_microbatches()

        num_opt_step = self.cfg.algorithm.n_minibatches
        self._batch_buffer_for_metrics.clear()

        training_metrics_list = []
        for _ in range(num_opt_step):
            self.log_info(f"[debug-hjh] rank={self._rank}, iter={_}, self.resharding_state={self.resharding_state}")
            if self.resharding_state:
                batch = self.batch_iterator
                training_metrics = self.training_step(batch, pipeline_func=True)
                training_metrics_list.append(training_metrics)
            # if not self.check_resharding(is_end=False):
            #     continue
        
            self.sync_with_scheduler()


        # self.check_resharding(is_end=True)

        rollout_metrics_valid_dp_group = self.process_rollout_metrics_with_elastic()
        if rollout_metrics_valid_dp_group is None:
            return
        rollout_metrics = RolloutResult.to_metrics(self._batch_buffer_for_metrics)
        rollout_metrics = compute_rollout_metrics_pipeline(
            rollout_metrics, self.cfg.data.max_prompt_length, self.response_len, data_parallel_group=rollout_metrics_valid_dp_group
        )
        self.average_respone_len = rollout_metrics["response_length"]

        return rollout_metrics, training_metrics_list

    # Elastic-Training
    def process_rollout_metrics_with_elastic(self):
        if not self.use_auto_scheduler:
            return parallel_state.get_data_parallel_group()

        max_data_parallel_group = parallel_state.get_data_parallel_group_elastic_max()
        max_data_parallel_ranks = torch.distributed.get_process_group_ranks(max_data_parallel_group)
        
        self.log_info(f"rank={self._rank}, max_data_parallel_ranks={max_data_parallel_ranks}")

        rollout_metrics_dp_ranks_states = torch.tensor([(dp_rank == self._rank and len(self._batch_buffer_for_metrics) > 0) for dp_rank in max_data_parallel_ranks]).cuda()
        torch.distributed.all_reduce(rollout_metrics_dp_ranks_states, torch.distributed.ReduceOp.MAX, group=max_data_parallel_group)

        rollout_metrics_dp_ranks_states = rollout_metrics_dp_ranks_states.tolist()
        rollout_metrics_valid_dp_ranks = [rank for rank, state in zip(max_data_parallel_ranks, rollout_metrics_dp_ranks_states) if state]

        if len(self._batch_buffer_for_metrics) > 0:
            return parallel_state.get_group_by_ranks(rollout_metrics_valid_dp_ranks)
        return None

    def calc_num_microbatches(self):
        if not self.resharding_state:
            return
        configure_batch_sizes(
                rank=torch.distributed.get_rank(),
                mbs=self.cfg.actor.micro_batch_size,
                gbs=self.cfg.actor.global_batch_size,
                dp=parallel_state.get_data_parallel_world_size(),
            )
        self.log_info(f"run_training_pipeline: mbs={self.cfg.actor.micro_batch_size}, gbs={self.cfg.actor.global_batch_size}, dp={parallel_state.get_data_parallel_world_size()} {get_num_microbatches()=}")

    def init_trainer_resharding(self, first_world_size:int = -1):
        """Init resharding func.
        """
        from rlinf.scheduler.dynamic_scheduler.util import (
            get_valid_dp_sizes
        )
        assert HAVE_RESHARDING, "params_resharding is not installed"
        
        
        args = get_args()
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        args.data_parallel_size = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size)
        args.load = None
        self.default_parallel_strategy = {
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "context_parallel_size": args.context_parallel_size
        }
        default_model_parallel_size_with_cp = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
        
        assert args.world_size == self.component_placement._cluster_num_gpus, "In Auto-Scheduler mode, actor should be initialized on all GPUs."
        assert self.component_placement.actor_world_size < args.world_size
        
        
        valid_dp_sizes = get_valid_dp_sizes(self.cfg, default_model_parallel_size_with_cp)
        assert len(valid_dp_sizes) > 0
        resharding_strategies = []
        
        for valid_dp_size in reversed(valid_dp_sizes):
            world_size = default_model_parallel_size_with_cp * valid_dp_size
            assert world_size <= self.component_placement._cluster_num_gpus
            
            resharding_strategies.append(
                {
                    "world_size" : world_size,
                    "tensor_model_parallel_size": args.tensor_model_parallel_size,
                    "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
                    "context_parallel_size": args.context_parallel_size
                }
            )
        
        assert resharding_strategies[0]['world_size'] == args.world_size
            
        self.trainer_resharding_func : Callable = resharding_init(
            model = self.model,
            optimizer = self.optimizer,
            opt_param_scheduler = self.lr_scheduler,
            trainer_parallel_strategies = resharding_strategies,
            offload_frist_strategy = False,
            model_provider = self.model_provider_func,
        )
        
        if first_world_size == -1:
            first_world_size = self.component_placement.actor_world_size
        
        self.resharding_state = True
        self.apply_parallel_strategy({'world_size':first_world_size})

    def apply_parallel_strategy(self, parallel_strategy):
        """Apply specified training parallel strategy
        """
        args = get_args()
        args.load = None
        
        parallel_keys = ["world_size", "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"]
        assert parallel_strategy.get("world_size") is not None, f"Error : can't find world_size in parallel_strategy"
        if parallel_strategy.get('context_parallel_size') is not None:
            assert parallel_strategy['context_parallel_size'] == self.default_parallel_strategy['context_parallel_size'], "change context_parallel_size is not supported"
            
        new_parallel_strategy = {}
        for parallel_key in parallel_keys:
            if parallel_strategy.get(parallel_key) is not None:
                new_parallel_strategy[parallel_key] = parallel_strategy[parallel_key]
            else:
                new_parallel_strategy[parallel_key] = self.default_parallel_strategy[parallel_key]
        
        self._logger.info(f"[ElasticMegatron-Info] rank={torch.distributed.get_rank()} start resharing with new_parallel_strategy = {new_parallel_strategy}")
        training_states, _ = self.trainer_resharding_func(new_parallel_strategy=new_parallel_strategy)
        
        if training_states is not None:
            self.model = training_states.model
            self.optimizer = training_states.optimizer
            self.lr_scheduler = training_states.opt_param_scheduler
            self.resharding_state = True
        else:
            self.resharding_state = False

    # Elastic-Training TODO(Chunyang)
    def broadcast_obj(self, obj):
        # device = torch.device("cpu")
        device = torch.cuda.current_device()
        if self._rank == 0:
            torch.distributed.broadcast_object_list([obj], src=0, device=device)
        else:
            obj_list = [None]
            torch.distributed.broadcast_object_list(obj_list, src=0, device=device)
            obj = obj_list[0]
        return obj

    def refresh_resharding_state(self, is_end: bool):
        parallel_state.barrier_with_gloo()
        resharding_resp = None
        if self._rank == 0:
            # Get batch_iterator data size information
            current_batch_size = 0
            micro_batch_counter = 0
            if hasattr(self, 'batch_iterator') and self.batch_iterator is not None:
                current_batch_size = self.batch_iterator.get_current_batch_size()
                micro_batch_counter = self.batch_iterator.get_micro_batch_counter()

            if not is_end:
                self.schedule_channel.put(
                    TrainerReq(
                        type='req_loop',
                        current_batch_size=current_batch_size,
                        micro_batch_counter=micro_batch_counter
                    ).serialize(),
                    queue_name=self.schedule_req_queue_name,
                    async_op=False,
                )
                self.log_info(f"Sent req_loop with batch info: current_batch_size={current_batch_size}, micro_batch_counter={micro_batch_counter}")
            else:
                self.schedule_channel.put(
                    TrainerReq(
                        type='req_end',
                        current_batch_size=current_batch_size,
                        micro_batch_counter=micro_batch_counter
                    ).serialize(),
                    queue_name=self.schedule_req_queue_name,
                    async_op=False,
                )
                self.log_info(f"Sent req_end with batch info: current_batch_size={current_batch_size}, micro_batch_counter={micro_batch_counter}")
            resharding_resp = self.schedule_channel.get(
                queue_name=self.schedule_resp_trainer_queue,
                async_op=False,
            )
            self.log_info(f"refresh_resharding_state: {resharding_resp}")
        resharding_resp = TrainerResp.deserialize(self.broadcast_obj(resharding_resp))
        # print(f"zcy_dbg [{torch.distributed.get_rank()}]: GRPOTask get resharding_resp: {resharding_resp}", flush=True)
        if not resharding_resp.need_reshard:
            assert self.resharding_state in (ReshardingState.RUN, ReshardingState.CONTINUE), f"Invalid resharding state: {self.resharding_state}"
            return None
        else:
            self.resharding_state = ReshardingState.RESHARDING
        return resharding_resp

    def check_resharding(self, is_end: bool):
        if not self.use_schedule:
            return True
        resharding_resp = self.refresh_resharding_state(is_end)
        assert self.resharding_state in (ReshardingState.RESHARDING, ReshardingState.RUN, ReshardingState.CONTINUE), f"Invalid resharding state: {self.esharding_state}"

        if self.resharding_state == ReshardingState.RESHARDING:
            self.log_info(f"rank [{self._rank}]: resharding start")
            self.apply_parallel_strategy(self.tag_to_state_id[resharding_resp.tag])
            if self._rank == 0:
                if not is_end:
                    # Get batch_iterator data size information
                    current_batch_size = 0
                    micro_batch_counter = 0
                    if hasattr(self, 'batch_iterator') and self.batch_iterator is not None:
                        current_batch_size = self.batch_iterator.get_current_batch_size()
                        micro_batch_counter = self.batch_iterator.get_micro_batch_counter()

                    self.schedule_channel.put(
                        TrainerReq(
                            type='resharded',
                            input_qsize=self._data_channel.qsize(queue_name=self._queue_name),
                            current_batch_size=current_batch_size,
                            micro_batch_counter=micro_batch_counter
                        ).serialize(),
                        queue_name=self.schedule_req_queue_name,
                        async_op=False,
                    )
                    self.log_info(f"Sent resharded with batch info: current_batch_size={current_batch_size}, micro_batch_counter={micro_batch_counter}, input_qsize={self._data_channel.qsize(queue_name=self._queue_name)}")
            self.calc_num_microbatches()
            report_device_info(f"zcy_dbg: rank [{self._rank}]: Actor after resharding:")

        if is_end and self._rank == 0:
            # Get batch_iterator data size information
            current_batch_size = 0
            micro_batch_counter = 0
            if hasattr(self, 'batch_iterator') and self.batch_iterator is not None:
                current_batch_size = self.batch_iterator.get_current_batch_size()
                micro_batch_counter = self.batch_iterator.get_micro_batch_counter()

            self.schedule_channel.put(
                TrainerReq(
                    type='loop_finish',
                    current_batch_size=current_batch_size,
                    micro_batch_counter=micro_batch_counter
                ).serialize(),
                queue_name=self.schedule_req_queue_name,
                async_op=False,
            )
        return self.resharding_state == ReshardingState.RUN

    # Inference
    @torch.no_grad()
    def inference_step(self, batch):
        set_eval(self)

        return self.run_forward_backward(batch, forward_only=True)

    def compute_logprobs(self):
        """Compute the log probabilities for the rollout batches."""
        batch = {
            "input_ids": self.rollout_batches["input_ids"],
            "attention_mask": self.rollout_batches["attention_mask"],
            "position_ids": self.rollout_batches["position_ids"],
        }
        self.rollout_batches["prev_logprobs"] = self.inference_step(batch)

    def compute_ref_logprobs(self):
        """Compute the reference log probabilities for the rollout batches."""
        assert self.ref_policy_state_dict is not None
        with cpu_weight_swap(self.model[0], self.ref_policy_state_dict):
            batch = {
                "input_ids": self.rollout_batches["input_ids"],
                "attention_mask": self.rollout_batches["attention_mask"],
                "position_ids": self.rollout_batches["position_ids"],
            }
            self.rollout_batches["ref_logprobs"] = self.inference_step(batch)

    def _setup_inference_weight_dst_ranks(self):
        self._weight_dst_rank_in_inference = self.get_inference_weight_dst_ranks(
            self.cfg.inference.model.tensor_model_parallel_size,
            self.cfg.inference.model.pipeline_model_parallel_size,
        )

    def get_inference_weight_dst_ranks(self, inference_tp, inference_pp):
        """
        Calculate the list of ranks corresponding to the first complete inference model parallel group after resharding.

        Returns:
            List of ranks for the first complete inference model parallel group after resharding
        """

        model_parallel_size = inference_tp * inference_pp
        # After resharding, the number of GPUs in a complete model parallel group = new TP × new PP
        # The first complete model parallel group consists of consecutive ranks starting from 0
        return list(range(model_parallel_size))

    def _get_inference_model_state_dict(self):
        """Get the state dictionary of the model for rollout."""
        return self.inference_weights_reshard.gather_and_reshard_model(
            unwrap_model(self.model)
        )

    def sync_model_to_inference(self):
        if not self.resharding_state:
            return
        inference_state_dict = self._get_inference_model_state_dict()

        for rank in self._weight_dst_rank_in_inference:
            if self._rank == rank:
                self.send(inference_state_dict, self.cfg.inference.group_name, rank)

        self.log_info(
            f"{self.__class__.__name__}: sync_model_to_inference resharding done."
        )

    # Advantages and returns
    def compute_advantages_and_returns(self):
        """Compute the advantages and returns for the rollout batches."""
        clear_memory()
        assert self.rollout_batches is not None
        mask = self.rollout_batches["attention_mask"][:, -self.response_len :]
        advantages, returns = calculate_adv_and_returns(
            self.cfg.algorithm.adv_type,
            self.rollout_batches["reward_scores"].cuda(),
            mask.cuda(),
            self.cfg.algorithm.group_size,
        )

        if self.cfg.algorithm.normalize_advantages:
            advantages = masked_normalization(advantages, mask)
        self.rollout_batches["advantages"] = advantages

        rollout_metrics, total_prompt_lengths, total_decode_lengths = (
            compute_rollout_metrics(
                self.rollout_batches, self.cfg.data.max_prompt_length, self.response_len
            )
        )

        rollout_metrics = cpu_dict(rollout_metrics)

        self.rollout_batches.pop("reward_scores")

        if self.cfg.actor.get("calculate_flops", False):
            total_generation_tflops = (
                self.flops_calculator.flops_generate(
                    total_prompt_lengths, total_decode_lengths
                )
                .float()
                .sum()
                .item()
                / 1e12
            )
            total_inference_tflops = (
                self.flops_calculator.flops_inference(
                    total_prompt_lengths + total_decode_lengths
                )
                .float()
                .sum()
                .item()
                / 1e12
            )

            rollout_metrics.update(
                {
                    "generation_tflops": total_generation_tflops,
                    "inference_tflops": total_inference_tflops,
                    "training_tflops": total_inference_tflops * 3,  # factor
                }
            )

        if self.cfg.actor.get("enable_dp_load_balance", False):
            self.rollout_batches = RolloutDataBalance.from_rollout_batches(
                rollout_batches=self.rollout_batches,
                dp_world_size=parallel_state.get_data_parallel_world_size(),
                dp_rank=parallel_state.get_data_parallel_rank(),
                dp_group=parallel_state.get_data_parallel_group(),
                partitioning_tool=get_seqlen_balanced_partitions,
            )

        return rollout_metrics

    # Rollout
    def _get_rollout_model_state_dict(self):
        """Get the state dictionary of the model for rollout."""
        return self.rollout_weights_reshard.gather_and_reshard_model(
            unwrap_model(self.model)
        )

    def _setup_rollout_weight_dst_ranks(self):
        """Setup destination ranks for token and weight communication."""
        if self.component_placement._placement_mode == PlacementMode.COLLOCATED:
            self._weight_dst_rank_in_rollout = (
                HybridRankMapper.get_actor_rank_to_rollout_rank(
                    self._rank,
                    self.component_placement.actor_tp_size,
                    self.component_placement.actor_pp_size,
                    self.component_placement.rollout_tp_size,
                    self.component_placement.actor_world_size,
                )
            )
        else:
            assert (
                self.component_placement._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            ), "Unsupported placement mode for token and weight destination ranks."
            self._weight_dst_rank_in_rollout = (
                DisaggRankMapper.get_actor_rank_to_rollout_ranks(
                    self.component_placement.actor_tp_size,
                    self.component_placement.actor_pp_size,
                    self.component_placement.actor_world_size,
                    self.component_placement.rollout_tp_size,
                    self.component_placement.rollout_world_size,
                )[self._rank]
            )
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self):
        if hasattr(self, "reshard_state_dict"):
            del self.reshard_state_dict

    def sync_model_to_rollout(self):
        if not self.resharding_state:
            return
        """Send the model weights to the destination ranks in the rollout task."""
        if self.component_placement._placement_mode == PlacementMode.COLLOCATED:
            if self.offload_optimizer:
                self.offload_megatron_optimizer()
            self.reshard_state_dict = self._get_rollout_model_state_dict()
            if self.offload_weight:
                self.offload_model_weights_and_grad(offload_grad=self.offload_grad)

            handle = {k: reduce_tensor(v) for k, v in self.reshard_state_dict.items()}
            self.send(
                handle, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )
        elif not self.use_pre_process_policy:
            assert (
                self.component_placement._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            ), "Unsupported placement mode for sending weights."
            assert isinstance(self._weight_dst_rank_in_rollout, list), (
                f"In disaggregated mode, weight_dst_rank_in_rollout should be a list of ranks, got {type(self._weight_dst_rank_in_rollout)}"
            )
            self.reshard_state_dict = self._get_rollout_model_state_dict()
            for weight_dst_rank in self._weight_dst_rank_in_rollout:
                self.send(
                    self.reshard_state_dict,
                    self._rollout_group_name,
                    weight_dst_rank,
                )
        else:
            assert (
                self.component_placement._placement_mode == PlacementMode.AUTO
            ), "Unsupported placement mode for sending weights."
            torch.cuda.synchronize()
            self.reshard_state_dict = self._get_rollout_model_state_dict()
            torch.cuda.synchronize()
            
            self.cuda_info("before offload")
            self.offload_model_weights_and_grad(offload_grad=True)
            self.cuda_info("after offload model")
            self.offload_megatron_optimizer()
            self.cuda_info("after offload opt")
                
            for weight_dst_rank in self._weight_dst_rank_in_rollout:
                self.send(
                    self.reshard_state_dict,
                    self._rollout_group_name,
                    weight_dst_rank,
                )
    
    def cuda_info(self, text: str = ""):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        free_gpu_memory /= 2**30
        total_gpu_memory /= 2**30

        memory_allocated = torch.cuda.memory_allocated() / 2**30
        memory_reserved = torch.cuda.memory_reserved() / 2**30

        self._logger.info(f"{text} {memory_allocated=:.2f} GiB, {memory_reserved=:.2f} GiB, {free_gpu_memory=:.2f} GiB, {total_gpu_memory=:.2f} GiB")
            

    def _get_rollout_result(self, rollout_channel: Channel):
        """Receive rollout results."""
        num_results_per_actor_dp = (
            self.component_placement.rollout_dp_size
            // self.component_placement.actor_dp_size
        )

        # Retrieve rollout results
        if self._is_data_io_rank:
            rollout_results = []
            for _ in range(num_results_per_actor_dp):
                # Each result is a list of RolloutResult because of rollout_batch_size_per_gpu
                rollout_result = rollout_channel.get()
                rollout_results.append(rollout_result)
        else:
            rollout_results = [None] * num_results_per_actor_dp

        # Broadcast in MP
        # NOTE: Use CPU broadcast to avoid NCCL's SM contention with rollout
        rollout_results = self.broadcast(rollout_results, ranks=self._mp_group_ranks)

        # Broadcast in CP
        context_parallel_src_rank = parallel_state.get_context_parallel_global_ranks()[
            0
        ]
        torch.distributed.broadcast_object_list(
            rollout_results,
            src=context_parallel_src_rank,
            group=parallel_state.get_context_parallel_group(),
        )

        # Flatten rollout results
        rollout_results = [
            rollout_result
            for rollout_dp_results in rollout_results
            for rollout_result in rollout_dp_results
        ]
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        self.log_debug(f"Received {rollout_result.num_sequence} responses")

        return rollout_result

    def process_rollout_result(self, input_channel: Channel):
        rollout_result = self._get_rollout_result(input_channel)
        self.rollout_batches = rollout_result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )

        if not self.cfg.reward.use_reward_model:
            self.answers = rollout_result.answers

        if self.offload_weight:
            self.onload_model_weights_and_grad(load_grad=self.offload_grad)
        if self.offload_optimizer:
            self.onload_megatron_optimizer()

    # Reward
    def _compute_rewards(self, answers: List[str]):
        all_reward_scores = []
        texts = []
        for response, response_len in zip(
            self.rollout_batches["input_ids"],
            self.rollout_batches["response_lengths"],
        ):
            response = response[
                self.cfg.data.max_prompt_length : self.cfg.data.max_prompt_length
                + response_len
            ]
            texts.append(
                self.tokenizer.decode(response.tolist(), skip_special_tokens=True)
            )

        if torch.distributed.get_rank() == parallel_state.get_model_parallel_src_rank():
            rewards = self.reward_fn(texts, answers)
            reward_scores = [
                self.cfg.reward.reward_scale
                if reward == 1
                else -self.cfg.reward.reward_scale
                for reward in rewards
            ]
            all_reward_scores.extend(reward_scores)

        if len(all_reward_scores) > 0:
            new_all_rewards = []

            for response in all_reward_scores:
                if response is None:
                    response = 0.0
                new_all_rewards.append(response)

            all_reward_scores = torch.as_tensor(
                new_all_rewards,
                dtype=torch.float,
                device=torch.cuda.current_device(),
            ).view(-1, 1)
        all_reward_scores = (
            broadcast_tensor_within_mp(all_reward_scores).flatten().to("cpu")
        )

        self.rollout_batches.update({"reward_scores": all_reward_scores})

    def compute_rewards(self):
        if not self.cfg.reward.use_reward_model:
            self._compute_rewards(self.answers)
