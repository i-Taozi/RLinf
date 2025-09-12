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
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from rlinf.algorithms.math.verifier.verify import MathRewardModel, math_verify_call
from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import (
    CompletionInfo,
    RolloutRequest,
    RolloutResult,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.scheduler.placement import RolloutReq, RolloutResp
from rlinf.utils.placement import ComponentPlacement, PlacementMode
from rlinf.workers.rollout.sglang import Engine, io_struct
from rlinf.workers.rollout.utils import (
    print_sglang_outputs,
)


class MetaInfoStatsCollector:
    """Collector for SGLang meta_info statistics
    
    This collector is only initialized when enabled via configuration.
    Add the following parameters to your generation config section:
    
    generation:
      collect_meta_stats: true  # Enable meta_info statistics collection
      meta_stats_file: "custom_meta_stats.jsonl"  # Optional: custom output file
      async_meta_stats_file: "custom_async_meta_stats.jsonl"  # Optional: custom async output file  
      schedule_meta_stats_file: "custom_schedule_meta_stats.jsonl"  # Optional: custom schedule output file
    """

    def __init__(self, output_file: str = "sglang_meta_stats.jsonl"):
        self.output_file = output_file
        self.stats_buffer = []
        self.buffer_size = 100  # Write to file every 100 records

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else ".", exist_ok=True)

        # Initialize file with header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                f.write("")  # Create empty file

    def collect_batch_stats(self, outputs: List[Dict], batch_id: int) -> None:
        """Collect statistics from a batch of SGLang outputs

        Args:
            outputs: List of SGLang output dictionaries
            batch_id: Unique identifier for this batch
        """
        current_time = time.time()

        for req_idx, output in enumerate(outputs):
            try:
                # Extract meta_info
                meta_info = output.get('meta_info', {})

                # Extract the specific metrics you requested
                stats_record = {
                    'timestamp': current_time,
                    'batch_id': batch_id,
                    'request_id': f"batch_{batch_id}_req_{req_idx}",
                    'prompt_tokens': meta_info.get('prompt_tokens', None),
                    'completion_tokens': meta_info.get('completion_tokens', None),
                    'e2e_latency': meta_info.get('e2e_latency', None),
                    'ttft': meta_info.get('ttft', None),
                    # Additional useful meta_info fields (if available)
                    'finish_reason': meta_info.get('finish_reason', {}).get('type', None),
                    'total_tokens': (meta_info.get('prompt_tokens', 0) + meta_info.get('completion_tokens', 0)) if meta_info.get('prompt_tokens') is not None and meta_info.get('completion_tokens') is not None else None,

                    # Add any other meta_info fields that might be useful
                    'meta_info_keys': list(meta_info.keys()),  # For debugging/inspection
                }

                self.stats_buffer.append(stats_record)

            except Exception as e:
                # Log error but continue processing
                error_record = {
                    'timestamp': current_time,
                    'batch_id': batch_id,
                    'request_id': f"batch_{batch_id}_req_{req_idx}",
                    'error': str(e),
                    'output_keys': list(output.keys()) if isinstance(output, dict) else 'not_dict'
                }
                self.stats_buffer.append(error_record)

        # Write to file if buffer is full
        if len(self.stats_buffer) >= self.buffer_size:
            self._flush_to_file()

    def _flush_to_file(self) -> None:
        """Write buffered statistics to file"""
        if not self.stats_buffer:
            return

        with open(self.output_file, 'a') as f:
            for record in self.stats_buffer:
                f.write(json.dumps(record) + '\n')

        print(f"Written {len(self.stats_buffer)} records to {self.output_file}")
        self.stats_buffer = []

    def finalize(self) -> None:
        """Flush any remaining data and close"""
        self._flush_to_file()
        print(f"Finalized stats collection. Data saved to {self.output_file}")

class SGLangWorker(Worker):
    def __init__(self, config: DictConfig, placement: ComponentPlacement):
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

    def _validate_weight_at_first(self):
        """
        Run a test prompt batch and print its output.
        """
        validate_sampling_params = {"temperature": 0, "max_new_tokens": 32}
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        if self._cfg.rollout.detokenize:
            outputs = self._engine.generate(prompts, validate_sampling_params)
            for prompt, output in zip(prompts, outputs):
                print("===============================")
                print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        else:
            prompt_ids = self._tokenizer(prompts).input_ids
            outputs = self._engine.generate(
                input_ids=prompt_ids, sampling_params=validate_sampling_params
            )
            print_sglang_outputs(prompts, outputs, self._tokenizer)
        print("===============================", flush=True)

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
            enable_torch_compile=self._cfg.rollout.use_torch_compile,
            torch_compile_max_bs=min(
                self._cfg.rollout.torch_compile_max_bs,
                self._cfg.rollout.max_running_requests,
            ),
            load_format="dummy" if not self._cfg.rollout.validate_weight else "auto",
            # disable_overlap_schedule=True,
            dtype=torch_dtype_from_precision(self._cfg.actor.model.precision),
            # sglang will only return text/output_ids when skip_tokenizer_init=False/True
            # text is not needed in RL training, so set to True can save time.
            skip_tokenizer_init=not self._cfg.rollout.detokenize,
            # sglang will print statistics every decode_log_interval decode steps.
            decode_log_interval=self._cfg.rollout.sglang_decode_log_interval,
            attention_backend=self._cfg.rollout.attention_backend,
            log_level="info",
            max_running_requests=self._cfg.rollout.max_running_requests,
            dist_init_addr=f"127.0.0.1:{str(Cluster.find_free_port())}",
        )

        self.log_on_first_rank(f"{server_args=}")

        self._engine = Engine(
            parent_address=self.worker_address,
            placement=self._placement,
            config=self._cfg,
            dp_rank=self._rank,
            **dataclasses.asdict(server_args),
        )

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

    def _stop(self):
        self.log_debug(
            f"[LLM dp {self._rank}] Received None input tokens, rollout end."
        )
        self._engine.offload_model_weights()

    def init_worker(self):
        # init rollout engine.
        self._init_engine()
        if self._cfg.rollout.validate_weight:
            self._validate_weight_at_first()

        # Rollout Engine should use parameters from actor, so it offloads its parameter first.
        self._engine.offload_model_weights()

    def sync_model_from_actor(self):
        self._engine.sync_hf_weight()

    def rollout(self, input_channel: Channel, output_channel: Channel):
        while True:
            request: RolloutRequest = input_channel.get()

            # Check if rollout has ended
            if request is None:
                self._stop()
                break

            # Repeat prompts based on the group_size config
            requests = request.repeat_and_split(self._rollout_batch_size)

            rollout_results = []
            for request in requests:
                # Generate outputs using the SGLang engine.
                results = self._engine.generate(
                    input_ids=request.input_ids,
                    sampling_params=self._sampling_params,
                    return_logprob=self._return_logprobs,
                )

                # Create RolloutResult from the outputs.
                rollout_result = RolloutResult.from_engine_results(
                    results, request.input_ids, request.answers, self._return_logprobs
                )
                rollout_results.append(rollout_result)

                # Put and print results
                if self._cfg.rollout.print_outputs:
                    prompts = self._tokenizer.batch_decode(request.input_ids)
                    print_sglang_outputs(prompts, results, self._tokenizer)
            output_channel.put(rollout_results)


def all_floats_equal(float_list: list[float], epsilon: float = 1e-9) -> bool:
    if len(float_list) <= 1:
        return True
    return np.std(float_list) < epsilon

class AsyncTaskQueue:
    def __init__(self):
        self.done_queue: asyncio.Queue[tuple[str, asyncio.Future[Any]]] = asyncio.Queue()
        self.doing_set = set()
        self.event_loop = asyncio.events.get_event_loop()
        self.exit_flag = False
        self._task_completion_event = asyncio.Event()

    def _on_completion(self, f):
        self.doing_set.remove(f)
        self.done_queue.put_nowait(f)

    def put_one(self, f, *, tag=None):
        if not asyncio.futures.isfuture(f) and not asyncio.coroutines.iscoroutine(f):
            raise TypeError(f"expect of future, not {type(f).__name__}")

        f = asyncio.ensure_future(f, loop=self.event_loop)
        self.doing_set.add(f)
        f.add_done_callback(self._on_completion)

    def __aiter__(self):
        return self

    async def __anext__(self):
        f = None
        while f == None:
            if self.exit_flag and self.unfinished_len() == 0:
                raise StopAsyncIteration
            f = await self.done_queue.get()
        return f.result()

    def unfinished_len(self):
        return len(self.doing_set) + self.done_queue.qsize()

    def set_exit_flag(self):
        self.exit_flag = True
        self.done_queue.put_nowait(None)

    async def loop_tasks(self, fn):
        async for args in self:
            fn(*args)
        self._task_completion_event.set()


class AsyncSGLangWorker(SGLangWorker):
    def __init__(self, config: DictConfig, placement: ComponentPlacement):
        super().__init__(config, placement)
        self._current_request: RolloutRequest = None
        self._input_queue = asyncio.Queue[RolloutRequest]()
        # (req_input_token_ids, sglang_result)
        self._output_queue = asyncio.Queue[Tuple[int, List[int], Dict]]()

        # Queue for completed rollouts
        self._completed_queue = asyncio.Queue[RolloutResult]()
        self._completion_info = CompletionInfo()
        self._rollout_end_event = asyncio.Event()
        self._sync_weight_end_event = asyncio.Event()

        self._reward_model = MathRewardModel(scale=self._cfg.reward.reward_scale)

        self._channel = self.connect_channel(self._cfg.rollout.channel.name)

        # all sglang dp will get input from the same queue, and put the results to another same queue.
        self._input_queue_name = self._cfg.rollout.channel.queue_name
        self._output_queue_name = self._cfg.rollout.channel.output_queue_name
        assert self._rollout_batch_size is None, (
            "rollout_batch_size_per_gpu is not supported in AsyncSGLangWorker"
        )

        # Initialize meta_stats_collector for async operations
        self.collect_meta_stats = getattr(self._cfg.rollout, 'collect_meta_stats', False)
        if self.collect_meta_stats:
            async_stats_file = getattr(self._cfg.rollout, 'async_meta_stats_file', f"sglang_meta_stats_async_rank_{self._rank}.jsonl")
            self.async_meta_stats_collector = MetaInfoStatsCollector(async_stats_file)
            self.async_batch_counter = 0

        
        self.use_schedule = False # self._cfg.cluster.scheduler.use_schedule
        if self.use_schedule:
            self.schedule_group_name = 'SchedulerTask'
            self.schedule_channel = self.connect_channel(self._cfg.cluster.scheduler.schedule_channel_name)
            # warmup
            self.schedule_channel.put(None, async_op=False)
            self.schedule_channel.get(async_op=False)

            self.schedule_req_queue_name = self._cfg.cluster.scheduler.schedule_req_queue_name
            self.schedule_resp_rollout_queue = f'{self._cfg.cluster.scheduler.schedule_resp_rollout_queue_name}:{self._rank}'

            self.state_tag = 'idle' # 'idle' | 'running' | 'migrate_out' | 'offload'

            # Initialize meta_info statistics collector only if enabled in config
            if self.collect_meta_stats:
                schedule_stats_file = getattr(self._cfg.rollout, 'schedule_meta_stats_file', f"sglang_meta_stats_rank_{self._rank}.jsonl")
                self.schedule_meta_stats_collector = MetaInfoStatsCollector(schedule_stats_file)
                self.schedule_batch_counter = 0

            # Initialize report progress
            self.progress_report_interval = 1.0  # seconds
            self.last_progress_report = time.time()
        
        self.use_auto_scheduler = (self._placement._placement_mode == PlacementMode.AUTO)
        
        if self.use_auto_scheduler:
            from rlinf.scheduler.dynamic_scheduler.util import (
                get_scheduler_channel,
                get_scheduler_request_queue,
                get_scheduler_response_queue,
            )
            self.schedule_channel = self.connect_channel(get_scheduler_channel('rollout'))            
            # warmup
            self.schedule_channel.put(None, async_op=False)
            self.schedule_channel.get(async_op=False)
        
            self.scheduler_request_queue = get_scheduler_request_queue(self._rank)
            self.scheduler_response_queue = get_scheduler_response_queue(self._rank)


    def init_worker(self):
        self._init_engine()

    def _calculate_reward_and_advantage(self, engine_results: List[Dict], answer: str):
        answers = [answer] * len(engine_results)
        texts: List[str] = []
        for res in engine_results:
            if hasattr(res, "text"):
                texts.append(res["text"])
            else:
                texts.append(
                    self._tokenizer.decode(res["output_ids"], skip_special_tokens=True)
                )

        rewards = self._reward_model.get_reward(texts, answers)
        rewards_tensor = torch.tensor(rewards)
        mean = rewards_tensor.mean()
        std = rewards_tensor.std()
        advantages = (rewards_tensor - mean) / (std + 1e-6)

        return rewards, advantages.tolist()

    async def _calculate_reward_and_advantage_processpool(
        self, engine_results: List[Dict], answer: str
    ):
        answers = [answer] * len(engine_results)
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

    async def _loop_progress_report(self, completion_info: CompletionInfo):
        """report progress to scheduler loop"""
        self.log_info('zcy_dbg: _loop_progress_report enter')
        while self.state_tag == 'running':
            await asyncio.sleep(self.progress_report_interval)
            progress_report = RolloutReq(
                type='report',
                rank=self._rank,
                total_requests=completion_info.num_requests,
                completed_requests=completion_info.num_completed,
                total_tasks=completion_info.num_requests * completion_info.n_result_each_request,
                completed_tasks=completion_info.n_result_each_request * completion_info.num_requests - sum(completion_info.n_result_each_request - len(i) for i in completion_info.results.values()),
                running_tasks=sum(completion_info.n_result_each_request - len(i) for i in completion_info.results.values()),
                timestamp=time.time()
            )
            self.log_info(f'zcy_dbg: gen progress_report={progress_report}')

            # send to scheduler
            await self.schedule_channel.put(
                progress_report.serialize(),
                queue_name=self.schedule_req_queue_name,
                async_op=True
            ).async_wait()

    async def _loop_progress_resp(self, task_queue: AsyncTaskQueue, completion_info: CompletionInfo):
        while True:
            self._logger.info('zcy_dbg: _loop_progress_resp: wait rollout_resp')
            rollout_resp = RolloutResp.deserialize(await self.schedule_channel.get(
                queue_name=self.schedule_resp_rollout_queue,
                async_op=True,
            ).async_wait())

            self.log_info(f'zcy_dbg: _loop_progress_resp: get rollout_resp={rollout_resp}')
            if rollout_resp.type == 'migrate_in':
                assert self.state_tag == 'running'
                for input_ids, results, abort_results, answer in zip(rollout_resp.input_ids, rollout_resp.results, rollout_resp.abort_results, rollout_resp.answers):
                    unique_id = completion_info.add_unique_id(input_ids, answer)
                    assert len(abort_results) != 0
                    assert len(results) + len(abort_results) == completion_info.n_result_each_request

                    for result in results:
                        completion_info.record_result(unique_id, result)

                    for abort_result in abort_results:
                        abort_input_ids = input_ids + abort_result["output_ids"]
                        task_queue.put_one(self._async_generate(completion_info, unique_id, abort_input_ids, self._sampling_params))

                await self.schedule_channel.put(
                    RolloutReq(
                        type='migrated_in',
                        rank=self._rank,
                        running_tasks=sum(completion_info.n_result_each_request - len(i) for i in completion_info.results.values()),
                    ).serialize(),
                    queue_name=self.schedule_req_queue_name,
                    async_op=True
                ).async_wait()

            elif rollout_resp.type == 'offload':
                assert self.state_tag == 'running'
                self.state_tag = 'offload'
                task_queue.set_exit_flag()
                break

            elif rollout_resp.type == 'migrate_out':
                assert self.state_tag == 'running'
                self.state_tag = 'migrate_out'
                _ =  await self._engine.tokenizer_manager.abort_generation()
                task_queue.set_exit_flag()
                break

            else:
                assert False
    
    async def _async_generate(
        self, completion_info: CompletionInfo, unique_id, input_ids, sampling_params: dict
    ):
        sampling_params = {k: v for k, v in sampling_params.items()}
        sampling_params['max_new_tokens'] -= len(input_ids) - len(completion_info.input_ids_map[unique_id])

        result = await self._engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
        )
        
        # fix the output_ids to the correct length if the generate was migrated from another rank.
        orig_input_ids_len = len(completion_info.input_ids_map[unique_id])
        if len(input_ids) != orig_input_ids_len:
            assert len(input_ids) > orig_input_ids_len
            result['output_ids'] = input_ids[orig_input_ids_len:] + result['output_ids']

        completion_info.record_result(unique_id, result)
        if completion_info.is_completed(unique_id):
            orig_input_ids, answer, results = completion_info.pop_results(unique_id)
            # rewards, advantages = await self._calculate_reward_and_advantage_processpool(
            #     results,
            #     answer,
            # )
            try:
                async with asyncio.timeout(100):
                    rewards, advantages = await asyncio.to_thread(
                        self._calculate_reward_and_advantage,
                        results,
                        answer,
                    )
            except TimeoutError:
                self.log_info(f"Timeout when calculating reward and advantage for unique_id={unique_id}. Using zero rewards and advantages.")
                rewards = [-1.0] * len(results)
                advantages = [0.0] * len(results)

            rollout_result = RolloutResult(
                num_sequence=len(results),
                prompt_lengths=[len(orig_input_ids)] * len(results),
                prompt_ids=[orig_input_ids] * len(results),
                response_lengths=[len(res["output_ids"]) for res in results],
                response_ids=[res["output_ids"] for res in results],
                is_end=[
                    res["meta_info"]["finish_reason"]["type"] == "stop"
                    for res in results
                ],
                rewards=rewards,
                advantages=advantages,
            )
            return result, rollout_result
        return result, None

    async def _put_result_to_output_queue(self, result: RolloutResult):
        await self._channel.put(
            item=result, queue_name=self._output_queue_name, async_op=True
        ).async_wait()

    def _gen_migrate_out_req(self, completion_info: CompletionInfo):
        unique_ids = [
            unique_id
            for unique_id, abort_results_list in completion_info.abort_results.items()
            if len(abort_results_list) != 0
        ]
        self.log_info(f'zcy_dbg: AsyncSGLangWorker._gen_migrate_out_req: unique_ids={unique_ids}')

        input_ids = []
        finished_results = []
        abort_results = []
        answers = []
        for unique_id in unique_ids:
            input_ids.append(completion_info.input_ids_map.pop(unique_id))
            finished_results.append(completion_info.results.pop(unique_id))
            abort_results.append(completion_info.abort_results.pop(unique_id))
            answers.append(completion_info.answers.pop(unique_id))

        return RolloutReq(
            type='migrated_out',
            rank=self._rank,
            input_ids=input_ids,
            results=finished_results,
            abort_results=abort_results,
            answers=answers,
        )


    async def abort_generation(self):
        """Abort the generation."""
        await self._engine.tokenizer_manager.abort_generation(
            obj=io_struct.AbortGenerationInput()
        )
    
         
    async def run_with_scheduler(self, task_queue: AsyncTaskQueue, completion_info: CompletionInfo):
        from rlinf.scheduler.dynamic_scheduler.util import (
                RolloutMigrateBatch,
                RolloutReport,
                RolloutAction,
                RolloutScheduleInfo,
            )
        
        async def report():
            self.log_info(f"[dev-hjh] rollout_{self._rank} start Report action")
            report = RolloutReport(
                total_requests=completion_info.num_requests,
                completed_requests=completion_info.num_completed,
                total_tasks=completion_info.num_requests * completion_info.n_result_each_request,
                completed_tasks=completion_info.n_result_each_request * completion_info.num_requests - sum(completion_info.n_result_each_request - len(i) for i in completion_info.results.values()),
                running_tasks=sum(completion_info.n_result_each_request - len(i) for i in completion_info.results.values()),
                timestamp=time.time()
            )
            scheduler_response = RolloutScheduleInfo(instance_id=self._rank, report=report) 
            
            await self.schedule_channel.put(scheduler_response, queue_name=self.scheduler_response_queue, async_op=True).async_wait()
            self.log_info(f"[dev-hjh] rollout_{self._rank} finish Report action")
        
        async def migrate_out(): 
            self.log_info(f"[dev-hjh] rollout_{self._rank} start Migrate_Out action")
            await self.abort_generation()
            task_queue.set_exit_flag()
            # wait async event
            await task_queue._task_completion_event.wait()
            
            self.log_info(f"[dev-hjh] rollout_{self._rank} finish self.abort_generation()")
            
            unique_ids = [unique_id for unique_id, abort_results_list in completion_info.abort_results.items() if len(abort_results_list) != 0]
            
            rollout_migrate_batches = []
            for unique_id in unique_ids:
                rollout_migrate_batches.append(
                    RolloutMigrateBatch(
                        input_ids=completion_info.input_ids_map.pop(unique_id),
                        results=completion_info.results.pop(unique_id),
                        abort_results=completion_info.abort_results.pop(unique_id),
                        answers=completion_info.answers.pop(unique_id))
                )
            
            scheduler_response = RolloutScheduleInfo(instance_id=self._rank, data=rollout_migrate_batches)
            await self.schedule_channel.put(scheduler_response, queue_name=self.scheduler_response_queue, async_op=True).async_wait()
            self.log_info(f"[dev-hjh] rollout_{self._rank} finish Migrate_Out action")
            
            
        async def migrate_in(scheduler_request : RolloutScheduleInfo):
            self.log_info(f"[dev-hjh] rollout_{self._rank} start Migrate_In action")
            rollout_migrate_batches = scheduler_request.data
            assert rollout_migrate_batches is not None
            for migrate_batch in rollout_migrate_batches:
                assert isinstance(migrate_batch, RolloutMigrateBatch)
                
                assert len(migrate_batch.abort_results) !=0
                assert len(migrate_batch.abort_results) + len(migrate_batch.results) == completion_info.n_result_each_request
                
                unique_id = completion_info.add_unique_id(migrate_batch.input_ids, migrate_batch.answers)
                
                for result in migrate_batch.results:
                    completion_info.record_result(unique_id, result)

                for abort_result in migrate_batch.abort_results:
                    abort_input_ids = migrate_batch.input_ids + abort_result["output_ids"]
                    task_queue.put_one(self._async_generate(completion_info, unique_id, abort_input_ids, self._sampling_params))
                    
            self.log_info(f"[dev-hjh] rollout_{self._rank} finish Migrate_In action")
        
        async def wait_for_finish():
            while True:
                await asyncio.sleep(0.1)
                running_tasks=sum(completion_info.n_result_each_request - len(i) for i in completion_info.results.values())
                if running_tasks == 0:
                    self.log_info(f'[dev-hjh] rollout{self._rank} wait_for_finish send finish flag')
                    task_queue.set_exit_flag()
                    break
        
        # Action with feedback : [RolloutAction.Report, RolloutAction.Migrate_Out]
        # Action without feedback : [RolloutAction.Migrate_In, RolloutAction.Finish]
        while True:
            request = await self.schedule_channel.get(queue_name=self.scheduler_request_queue, async_op=True).async_wait()
            self.log_info(f"[debug-hjh] rollout-{self._rank} recv request.action={request.action}")
            if request.action == RolloutAction.Report:
                await report()
            elif request.action == RolloutAction.Migrate_In:
                await migrate_in(request)
            elif request.action == RolloutAction.Migrate_Out:
                await migrate_out()
            elif request.action == RolloutAction.Wait_For_Finish:
                await wait_for_finish()
            elif request.action == RolloutAction.Finish:
                self.log_info(f"[dev-hjh] rollout_{self._rank} call set_exit_flag()")
                task_queue.set_exit_flag()

            if task_queue.exit_flag == True:
                running_tasks=sum(completion_info.n_result_each_request - len(i) for i in completion_info.results.values())
                assert running_tasks == 0, f"ready to break run_with_scheduler() but running_tasks={running_tasks}"
                self.log_info(f"[debug-hjh] rollout-{self._rank} break run_with_scheduler()")
                break



    async def rollout(self):
        self.log_info("Starting async generation...")

        completion_info = CompletionInfo(self._logger)
        task_queue = AsyncTaskQueue()

        rollout_request: RolloutRequest = await self._channel.get(
            queue_name=self._input_queue_name, async_op=True
        ).async_wait()
        generate_max_batch_size = rollout_request.n * len(rollout_request.input_ids)

        completion_info.n_result_each_request = rollout_request.n

        for input_ids, answer in zip(rollout_request.input_ids, rollout_request.answers):
            unique_id = completion_info.add_unique_id(input_ids, answer)
            for _ in range(rollout_request.n):
                task_queue.put_one(self._async_generate(completion_info, unique_id, input_ids, self._sampling_params))

        # Collect all results for meta_info statistics
        return_tasks = []
        all_results = []

        def post_process_result(result, rollout_result: RolloutResult):
            all_results.append(result)  # Collect for meta_info stats
            if rollout_result is not None:
                return_tasks.append(
                    asyncio.create_task(
                        self._put_result_to_output_queue(rollout_result)
                    )
                )
        loop_handle_rollout_tasks = asyncio.create_task(task_queue.loop_tasks(post_process_result))

        # wait for generate result
        if self.use_schedule:
            # start progress reporting
            # loop wont' be finished if unfinished_len is 0. loop will be finished only when state_tag is 'idle' and unfinished_len is 0.
            self.state_tag = 'running'
            loop_handle_report = asyncio.create_task(self._loop_progress_report(completion_info))
            loop_handle_resp = asyncio.create_task(self._loop_progress_resp(task_queue, completion_info))
            await asyncio.gather(loop_handle_rollout_tasks, loop_handle_report, loop_handle_resp)
        elif self.use_auto_scheduler:
            scheduler_tasks = asyncio.create_task(self.run_with_scheduler(task_queue, completion_info))
            # send_finish_flag_tasks = asyncio.create_task(self.send_finish_flag(completion_info))
            # await asyncio.gather(loop_handle_rollout_tasks, scheduler_tasks, send_finish_flag_tasks)
            await asyncio.gather(loop_handle_rollout_tasks, scheduler_tasks)
        else:
            # loop will be finished if unfinished_len is 0
            self.state_tag = 'idle'
            task_queue.set_exit_flag()
            await asyncio.gather(loop_handle_rollout_tasks)

        # wait for send result
        await asyncio.gather(*return_tasks)
        

        # Collect meta_info statistics for all results only if enabled in config
        if self.collect_meta_stats and hasattr(self, 'async_meta_stats_collector'):
            self.async_meta_stats_collector.collect_batch_stats(all_results, self.async_batch_counter)
            self.async_batch_counter += 1

        self.log_info(f"Async generation and send completed. send results len={len(return_tasks)}")
        await self.offload_engine()

        if self.use_schedule:
            assert self.state_tag in ['migrate_out', 'offload']
            if self.state_tag == 'migrate_out':
                await self.schedule_channel.put(
                    self._gen_migrate_out_req(completion_info).serialize(),
                    queue_name=self.schedule_req_queue_name,
                    async_op=True,
                ).async_wait()

            await self.offload_engine()
            self.log_info("Async offload completed.")
            self.state_tag = 'idle'

            await self.schedule_channel.put(
                RolloutReq(type='offloaded', rank=self._rank).serialize(),
                queue_name=self.schedule_req_queue_name,
                async_op=True,
            ).async_wait()

    async def offload_engine(self):
        """
        Offload the model weights from the SGLang engine.
        """
        await self._engine.tokenizer_manager.offload_model_weights(
            io_struct.OffloadReqInput()
        )

    async def sync_model_from_actor(self):
        """Update the weights of the SGLang engine."""
        await self._engine.tokenizer_manager.sync_hf_weight(
            obj=io_struct.SyncHFWeightInput()
        )

    def shutdown(self):
        """
        Shutdown the SGLang task.
        """
        self.log_info(f"Shutting down SGLang worker {self._rank} ...")

        # Finalize meta_info statistics collectors if they exist
        if hasattr(self, 'async_meta_stats_collector'):
            self.async_meta_stats_collector.finalize()
        
        if hasattr(self, 'schedule_meta_stats_collector'):
            self.schedule_meta_stats_collector.finalize()

        self._engine.shutdown()
        self.log_info(f"SGLang worker {self._rank} shutdown complete.")

