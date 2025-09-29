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

import logging

import torch
from omegaconf import DictConfig
from sglang.srt.managers.io_struct import (
    AbortReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler import Scheduler as _Scheduler
from sglang.srt.managers.scheduler import logger
from sglang.srt.managers.scheduler import (
    run_scheduler_process as _run_scheduler_process,
)

from rlinf.scheduler import Worker, WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import (
    RankMapper,
)

from .io_struct import (
    AbortGenerationInput,
    AbortGenerationOutput,
    SyncHFWeightInput,
    SyncHFWeightOutput,
    TaskMethodInput,
    TaskMethodOutput,
)

logger.setLevel(logging.WARNING)


class Scheduler(_Scheduler):
    """
    Overridden class of SGLang's TP worker class _Scheduler.
    A Scheduler is a Task that manages the TP worker, and performs necessary weight synchronization with actor and weight offloading.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `TpModelWorkerClient` is used when ServerArgs.enable_overlap=True, and it has 'worker' attribute.
        # But in early SGLang version, `TpModelWorker` doesn't have 'worker' attribute.
        if not hasattr(self.tp_worker, "worker"):
            self.tp_worker.worker = self.tp_worker

        self._request_dispatcher._mapping.extend(
            [
                (TaskMethodInput, self.run_task_method),
                (SyncHFWeightInput, self.sync_hf_weight),
                (AbortGenerationInput, self.abort_generation),
            ]
        )

        # it's important to use load_weight to load resharded weight from megatron
        for _, module in self.tp_worker.worker.model_runner.model.named_modules():
            if hasattr(module, "use_presharded_weights"):
                module.use_presharded_weights = True

        self.is_weight_offloaded = False

    def cuda_info(self, text: str = ""):
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        free_gpu_memory /= 2**30
        total_gpu_memory /= 2**30

        memory_allocated = torch.cuda.memory_allocated() / 2**30
        memory_reserved = torch.cuda.memory_reserved() / 2**30

        self._rlinf_worker.log_info(
            f"[dp {self._rlinf_worker.get_parent_rank()}-tp {self.tp_rank}] {text} "
            f"{memory_allocated=:.2f} GiB, {memory_reserved=:.2f} GiB, "
            f"{free_gpu_memory=:.2f} GiB, {total_gpu_memory=:.2f} GiB"
        )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert self.is_weight_offloaded is False, "Weight has been offloaded!"
        self.is_weight_offloaded = True
        return super().release_memory_occupation(recv_req)

    def sync_hf_weight(self, recv_req: SyncHFWeightInput):
        use_cudagraph = not self.cfg.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED

        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self._rlinf_worker.recv(
            src_group_name=self._actor_group_name,
            src_rank=self.actor_weight_rank,
        )

        model = self.tp_worker.worker.model_runner.model

        if self.is_weight_offloaded:
            self.resume_memory_occupation(ResumeMemoryOccupationReqInput())
            self.is_weight_offloaded = False

        if colocate:
            for name, handle in state_dict.items():
                func, args = handle
                list_args = list(args)
                # NOTE: the key is to change device id to the current device id
                # in case two processes have different CUDA_VISIBLE_DEVICES
                list_args[6] = torch.cuda.current_device()
                new_weight = func(*list_args)

                model.load_weights([(name, new_weight)])
                del new_weight
        else:
            # disaggregate mode, recv tensor directly
            for name, tensor in state_dict.items():
                model.load_weights([(name, tensor)])
        self.flush_cache()
        return SyncHFWeightOutput()

    def run_task_method(self, obj: TaskMethodInput):
        """
        Run a CommTask method with the given name and arguments.
        NOTE: will call wait() if async_op is True.
        """
        result = getattr(self, obj.method_name)(*obj.args, **obj.kwargs)
        if "async_op" in obj.kwargs and obj.kwargs["async_op"]:
            result = result.wait()
        return TaskMethodOutput(method_name=obj.method_name, result=result)

    def abort_request(self, recv_req: AbortReq):
        # Compared to the original SGLang implementation, we will remove all requests that start with the given rid.
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in sorted(to_del, reverse=True):
            req = self.waiting_queue.pop(i)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete requests in the running batch
        for req in self.running_batch.reqs:
            if req.rid.startswith(recv_req.rid) and not req.finished():
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_abort = True

    def abort_generation(self, recv_req: AbortGenerationInput):
        # clear waiting reqs
        waiting_reqs = []
        # waiting_reqs.append(self.waiting_queue)
        for req in self.waiting_queue:
            req.to_abort = True

        # abort every running req with no kvcache
        running_reqs = []
        running_reqs.append(self.running_batch.reqs)
        for req in self.running_batch.reqs:
            req.to_abort = True
        res = AbortGenerationOutput(
            waiting_reqs=waiting_reqs, running_reqs=running_reqs
        )
        return res

    def init_rlinf_worker(
        self,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        config: DictConfig,
    ):
        # WARNNING(wyq): Is world_size == self.tp_size when we enable EP in MoE?
        self._rlinf_worker = Worker(
            parent_address=parent_address, world_size=self.tp_size, rank=self.tp_rank
        )
        self.cfg = config
        self._actor_group_name = self.cfg.actor.group_name
        self.placement_mode = placement.placement_mode
        self.actor_weight_rank = RankMapper.get_rollout_rank_to_actor_rank_map(
            placement
        )[(self._rlinf_worker.get_parent_rank(), self._rlinf_worker._rank)]

        self._rlinf_worker.log_info(
            f"Running Scheduler dp rank {self._rlinf_worker.get_parent_rank()}, tp rank {self.tp_rank}, corresponding actor weight rank = {self.actor_weight_rank}"
        )

    def get_scheduler_running_state(self):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
        )
        num_running_reqs = len(self.running_batch.reqs)
        return {
            "num_running_reqs": num_running_reqs,
            "max_running_reqs": self.max_running_requests,
            "num_used_tokens": num_used,
            "max_total_num_tokens": self.max_total_num_tokens,
            "token_usage": num_used / self.max_total_num_tokens,
            "num_queue_reqs": len(self.waiting_queue),
        }


def run_scheduler_process(*args, **kwargs):
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "sglang.srt.managers.scheduler.Scheduler",
        "rlinf.hybrid_engines.sglang.common.sgl_scheduler.Scheduler",
    )
    Patcher.apply()
    _run_scheduler_process(*args, **kwargs)
