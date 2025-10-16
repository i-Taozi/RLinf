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

import asyncio
import dataclasses
from typing import Dict, List

import torch
from omegaconf import DictConfig
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
    SeqGroupInfo,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.scheduler.dynamic_scheduler.manager import RolloutScalingScheduler
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_scheduler_channel,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang import Engine, io_struct
from rlinf.workers.rollout.utils import (
    MetaInfoStatsCollector,
    RolloutEngineStats,
    RunningStatusManager,
    print_sglang_outputs,
)
from toolkits.math_verifier.verify import MathRewardModel, math_verify_call


class SGLangWorker(Worker):
    def __init__(self, config: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)

        self._cfg = config
        self._placement = placement

        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.rollout.model_dir)
        self._eos = self._cfg.rollout.eos or self._tokenizer.eos_token_id
        self._return_logprobs = self._cfg.rollout.return_logprobs
        self._sampling_params = self._get_sampling_param_from_config()
        if self._cfg.algorithm.rollout_batch_size_per_gpu is None:
            self._rollout_batch_size = None
        else:
            self._rollout_batch_size = (
                self._cfg.algorithm.rollout_batch_size_per_gpu
                * self._cfg.rollout.tensor_parallel_size
                * self._cfg.rollout.pipeline_parallel_size
            )

        self._validate_sampling_params = {"temperature": 0, "max_new_tokens": 32}
        self._validate_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        self._reward_model = MathRewardModel(scale=self._cfg.reward.reward_scale)

        self.status_manager = RunningStatusManager()

        # Initialize meta_stats_collector for async operations
        self._collect_meta_stats = getattr(
            self._cfg.rollout, "collect_meta_stats", False
        )
        self._use_auto_scheduler = self._placement.is_auto

        if self._collect_meta_stats:
            self._init_meta_stats_collector()
        if self._use_auto_scheduler:
            self._init_scheduler()

    def _init_scheduler(self):
        self.schedule_channel = self.connect_channel(get_scheduler_channel("rollout"))

        self._scheduler = RolloutScalingScheduler(
            self._rank, self.schedule_channel, self
        )

    def _init_meta_stats_collector(self):
        async_stats_file = getattr(
            self._cfg.rollout,
            "async_meta_stats_file",
            f"sglang_meta_stats_async_rank_{self._rank}.jsonl",
        )
        self.async_meta_stats_collector = MetaInfoStatsCollector(async_stats_file)
        self.async_batch_counter = 0

    def _collect_stats(self, engine_results: List[Dict]):
        self.async_meta_stats_collector.collect_batch_stats(
            engine_results, self.async_batch_counter
        )
        self.async_batch_counter += 1

    def _get_sampling_param_from_config(self) -> dict:
        """
        Get sampling parameters from the configuration.
        """
        cfg_sampling_params = self._cfg.algorithm.sampling_params
        if cfg_sampling_params.use_greedy:
            sampling_params = {
                "temperature": 0,
                "max_new_tokens": cfg_sampling_params.max_new_tokens,
            }
        else:
            sampling_params = {
                "temperature": cfg_sampling_params.temperature,
                "top_k": cfg_sampling_params.top_k,
                "top_p": cfg_sampling_params.top_p,
                "repetition_penalty": cfg_sampling_params.repetition_penalty,
                "max_new_tokens": cfg_sampling_params.max_new_tokens,
            }
        return sampling_params

    def _init_engine(self):
        use_cudagraph = not self._cfg.rollout.enforce_eager

        server_args = ServerArgs(
            model_path=self._cfg.rollout.model_dir,
            disable_cuda_graph=not use_cudagraph,
            cuda_graph_max_bs=min(
                self._cfg.rollout.cuda_graph_max_bs,
                self._cfg.rollout.max_running_requests,
            ),
            tp_size=self._cfg.rollout.tensor_parallel_size,
            mem_fraction_static=self._cfg.rollout.gpu_memory_utilization,
            enable_memory_saver=use_cudagraph,
            enable_torch_compile=self._cfg.rollout.sglang.use_torch_compile,
            torch_compile_max_bs=min(
                self._cfg.rollout.sglang.torch_compile_max_bs,
                self._cfg.rollout.max_running_requests,
            ),
            load_format="dummy" if not self._cfg.rollout.validate_weight else "auto",
            # disable_overlap_schedule=True,
            dtype=torch_dtype_from_precision(self._cfg.actor.model.precision),
            # sglang will only return text/output_ids when skip_tokenizer_init=False/True
            # text is not needed in RL training, so set to True can save time.
            skip_tokenizer_init=not self._cfg.rollout.detokenize,
            # sglang will print statistics every decode_log_interval decode steps.
            decode_log_interval=self._cfg.rollout.sglang.decode_log_interval,
            attention_backend=self._cfg.rollout.sglang.attention_backend,
            log_level="info",
            max_running_requests=self._cfg.rollout.max_running_requests,
            dist_init_addr=f"127.0.0.1:{str(Cluster.find_free_port())}",
        )

        self.log_on_first_rank(f"{server_args=}")
        self._engine = Engine(
            **dataclasses.asdict(server_args),
        )

    def _pre_process_rollout_request(
        self, request: RolloutRequest
    ) -> List[List[RolloutRequest]]:
        group_size = request.n
        repeated_request = request.repeat()
        if self._rollout_batch_size is not None:
            assert len(repeated_request.input_ids) % self._rollout_batch_size == 0, (
                f"rollout_batch_size {self._rollout_batch_size} must divide the total number of requests {len(repeated_request)}"
            )
            num_batch = len(repeated_request.input_ids) // self._rollout_batch_size
        else:
            num_batch = 1

        # Split the repeated request into smaller requests based on the rollout batch size
        # avoid too large request that may cause KV cache OOM
        split_requests = repeated_request.split(num_batch)
        if self._placement.is_disaggregated:
            num_prompts_per_request = len(split_requests[0].input_ids) // group_size
            # for disaggregated mode, split to ensure each small request has full group_size prompts
            return [r.split(num_prompts_per_request) for r in split_requests]
        else:
            return [r.split(1) for r in split_requests]

    def shutdown(self):
        """
        Shutdown the SGLang task.
        """
        # Finalize meta_info statistics collectors if they exist
        if self._collect_meta_stats:
            self.async_meta_stats_collector.finalize()

        self.log_info(f"Shutting down SGLang worker {self._rank} ...")
        self._engine.shutdown()
        self.log_info(f"SGLang worker {self._rank} shutdown complete.")

    async def _validate_weight_at_first(self):
        """
        Run a test prompt batch and print its output.
        """
        if self._cfg.rollout.detokenize:
            self.log_warning(
                "validate_weight with detokenize=True is not supported yet."
            )
        else:
            prompt_ids = self._tokenizer(self._validate_prompts).input_ids
            _, _, engine_results = await self._async_generate(
                prompt_ids, None, self._validate_sampling_params, False
            )
            print_sglang_outputs(
                self._validate_prompts, engine_results, self._tokenizer
            )
            print("===============================", flush=True)

    async def _stop(self):
        self.log_debug(
            f"[LLM dp {self._rank}] Received None input tokens, rollout end."
        )
        if not self._placement.is_disaggregated:
            await self.offload_engine()

    def _compute_reward_and_advantage(
        self, engine_results: List[Dict], answers: List[List[str]]
    ):
        texts: List[str] = []
        for res in engine_results:
            if hasattr(res, "text"):
                texts.append(res["text"])
            else:
                texts.append(
                    self._tokenizer.decode(res["output_ids"], skip_special_tokens=True)
                )

        results = math_verify_call(texts, answers)
        rewards = [(1 if r else -1) * self._reward_model.scale for r in results]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float)

        mean = rewards_tensor.mean()
        std = rewards_tensor.std()
        advantages = (rewards_tensor - mean) / (std + 1e-6)

        return rewards, advantages.tolist()

    async def _async_generate(
        self,
        input_ids: List[List[int]],
        answers: List[List[str]],
        sampling_params: dict,
        return_logprobs: bool,
    ):
        result = await self._engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprobs,
        )
        # SGLang does not return input_ids, so we need to pass them for further usage.
        return input_ids, answers, result

    async def init_worker(self):
        self._init_engine()
        await self._engine.tokenizer_manager.run_task_method(
            io_struct.TaskMethodInput(
                method_name="init_rlinf_worker",
                args=(
                    self.worker_address,
                    self._placement,
                    self._cfg,
                ),
            )
        )
        self.log_info(f"SGLang worker {self._rank} initialized.")
        if self._cfg.rollout.validate_weight:
            await self._validate_weight_at_first()
        if self._placement.is_collocated:
            await self.offload_engine()
        if self._use_auto_scheduler:
            asyncio.create_task(self._scheduler.main_loop())

    async def offload_engine(self):
        """
        Offload the model weights from the SGLang engine.
        """
        await self._engine.tokenizer_manager.release_memory_occupation(
            obj=ReleaseMemoryOccupationReqInput()
        )

    async def abort_generation(self):
        """Abort the generation."""
        await self._engine.tokenizer_manager.abort_generation(
            obj=io_struct.AbortGenerationInput()
        )

    async def sync_model_from_actor(self):
        """Update the weights of the SGLang engine."""
        await self._engine.tokenizer_manager.sync_hf_weight(
            obj=io_struct.SyncHFWeightInput()
        )

    async def check_running_state(self):
        state = await self._engine.tokenizer_manager.run_task_method(
            io_struct.TaskMethodInput(method_name="get_scheduler_running_state")
        )
        state = RolloutEngineStats(**state)

        return state

    async def _async_generate_single(
        self,
        idx: int,
        input_ids: List[int],
        sampling_params: dict,
        return_logprob: bool,
    ) -> tuple[int, Dict]:
        result = await self._engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
        )
        return idx, result

    async def _async_generate_group(self, seq_group_info: SeqGroupInfo):
        if seq_group_info.num_aborted > 0:
            # migrated sequences need to continue generation
            seq_idx = seq_group_info.idx_aborted.copy()
            seq_group_info.idx_aborted.clear()
            input_batch: List[List[int]] = []
            sampling_params: List[Dict] = []
            for idx in seq_idx:
                generated_ids: List[int] = seq_group_info.results[idx]["output_ids"]
                input_batch.append(seq_group_info.input_ids + generated_ids)
                params = self._sampling_params.copy()
                params["max_new_tokens"] -= len(generated_ids)
                sampling_params.append(params)
        else:
            # new sequence group
            assert seq_group_info.num_returned == 0
            seq_idx = list(range(seq_group_info.group_size))
            input_batch = [seq_group_info.input_ids] * seq_group_info.group_size
            sampling_params = [self._sampling_params] * seq_group_info.group_size

        tasks = [
            asyncio.create_task(
                self._async_generate_single(
                    idx, input_ids, params, self._return_logprobs
                )
            )
            for idx, input_ids, params in zip(seq_idx, input_batch, sampling_params)
        ]
        for future in asyncio.as_completed(tasks):
            idx, result = await future
            seq_group_info.record_sglang_result(idx, result, self._logger)

        return seq_group_info

    async def rollout(self, input_channel: Channel, output_channel: Channel):
        if self._rank == 0:
            self.log_info("Starting async generation...")
        request: RolloutRequest = input_channel.get()
        groups = request.to_seq_group_infos()
        async_wait_type = (
            asyncio.FIRST_COMPLETED
            if self._placement.is_pipeline
            else asyncio.ALL_COMPLETED
        )
        with output_channel.device_lock, self.worker_timer():
            num_residual = self.status_manager.num_seq_group
            assert num_residual == 0, (
                f"There are {num_residual} "
                f"sequence group{'' if num_residual == 1 else 's'} before rollout."
            )

            for group in groups:
                task = asyncio.create_task(self._async_generate_group(group))
                self.status_manager.add_task(group, task)

            all_rollout_results = []
            while pending := self.status_manager.get_running_tasks():
                done, pending = await asyncio.wait(pending, return_when=async_wait_type)
                returned_seq_groups: List[SeqGroupInfo] = [
                    task.result() for task in done
                ]
                for group in returned_seq_groups:
                    if group.all_completed:
                        rollout_result = RolloutResult.from_sglang_results(
                            group.results,
                            group.group_size,
                            [group.input_ids] * group.group_size,
                            [group.answer] * group.group_size,
                            self._return_logprobs,
                        )
                        # collocated mode will collect all results and send at once at the end
                        # pipeline mode will send result immediately
                        all_rollout_results.append(rollout_result)
                        if self._placement.is_pipeline:
                            (
                                rewards,
                                advantages,
                            ) = await asyncio.to_thread(
                                self._compute_reward_and_advantage,
                                group.results,
                                [group.answer] * group.group_size,
                            )

                            rollout_result.rewards = torch.tensor(
                                rewards, dtype=torch.float32
                            ).reshape(-1, 1)
                            rollout_result.advantages = advantages

                            await output_channel.put(
                                item=rollout_result, async_op=True
                            ).async_wait()
                        self.status_manager.mark_done(group)
                    else:
                        self.status_manager.mark_aborted(group)

                if (
                    self._use_auto_scheduler
                    and self.status_manager.num_seq_group_running == 0
                ):
                    # rollout should not exit immediately when using auto scheduler
                    # because there might be migrations
                    # if so, `pending` will not be empty in while loop condition
                    await self.status_manager.wait_notification()

            self.status_manager.clear()

            if self._collect_meta_stats:
                self._collect_stats(all_rollout_results)

            if not self._placement.is_pipeline:
                rollout_result = RolloutResult.merge_result_list(all_rollout_results)
                await output_channel.put(
                    item=rollout_result, async_op=True
                ).async_wait()

            if self._placement.is_collocated or self._placement.is_auto:
                await self.offload_engine()
                if self._use_auto_scheduler:
                    await self._scheduler.report_offloaded()
