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

from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.scheduler.dynamic_scheduler.manager import (
    ActorManager,
    ComponentManager,
    InferenceManager,
    RolloutManager,
)
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_valid_dp_sizes,
)
from rlinf.utils.placement import ComponentPlacement


class SchedulerWorker(Worker):
    """Dynamic Scheduler."""

    def __init__(self, config: DictConfig, component_placement: ComponentPlacement):
        """Initialize the SchedulerWorker."""
        super().__init__()
        self.cfg = config
        self.component_placement = component_placement
        self.components = self.component_placement._components
        self.total_gpus = self.component_placement._cluster_num_gpus

        assert self.cfg.rollout.rollout_backend == "sglang", (
            "only sglang is supported for dynamic scheduler"
        )
        assert self.cfg.actor.training_backend == "megatron", (
            "only megatron is supported for dynamic scheduler"
        )

        # Set policies for scheduler
        self.use_pre_process_policy = getattr(
            self.cfg.cluster, "use_pre_process_policy", True
        )
        self.use_wait_before_last_iter_policy = getattr(
            self.cfg.cluster, "use_wait_before_last_iter_policy", True
        )

        assert "rollout" in self.components, "rollout component is required"
        assert "actor" in self.components, "actor component is required"

        component_manager_kwargs = {
            "config": config,
            "component_placement": component_placement,
            "use_pre_process_policy": self.use_pre_process_policy,
            "use_wait_before_last_iter_policy": self.use_wait_before_last_iter_policy,
            "_logger": self._logger,
            "channel_factory": self.create_channel,
        }

        self.rollout_manager = RolloutManager(
            **component_manager_kwargs,
        )
        self.actor_manager = ActorManager(
            **component_manager_kwargs,
        )

        if "inference" in self.components:
            self.inference_manager = InferenceManager(
                **component_manager_kwargs,
            )
        else:
            self.inference_manager = ComponentManager(
                "dummy", None, None, None, None, None, None
            )

        actor_valid_dp_sizes = get_valid_dp_sizes(
            self.cfg,
            self.component_placement._cluster_num_gpus,
            self.actor_manager.model_parallel_size,
        )

        self.actor_manager.actor_valid_dp_sizes = actor_valid_dp_sizes
        self.rollout_manager.actor_valid_dp_sizes = actor_valid_dp_sizes

    async def run(self):
        """Run the scheduler."""
        await self.pre_process()
        await self.main_loop()

    async def pre_process(self):
        """Pre process.

        Reset component manager states and execute use_pre_process_policy.
        """
        await self.rollout_manager.pre_process()
        await self.actor_manager.pre_process()
        await self.inference_manager.pre_process()

    async def main_loop(self):
        """Main loop.

        The signal synchronization granularity is consistent with the actor training granularity.
        Resource allocation is performed after the actor has completed one execution of optimizer.step().
        """
        available_gpu_num = 0
        for train_iter in range(self.cfg.algorithm.n_minibatches):
            # Wait for actor ready to update
            await self.actor_manager.wait_for_actor_update()

            # Trying to release the resource of rollout and inference
            rollout_released_gpu_num = await self.rollout_manager.release_resource(
                train_iter,
                self.actor_manager.current_instance_num,
            )
            inference_released_gpu_num = await self.inference_manager.release_resource(
                train_iter, self.rollout_manager.current_instance_num == 0
            )
            self.log_info(
                f"[Release-Info] train_iter={train_iter}, rollout_released_gpu_num={rollout_released_gpu_num}, inference_released_gpu_num={inference_released_gpu_num}, available_gpu_num={available_gpu_num}"
            )
            total_available_gpu_num = (
                rollout_released_gpu_num
                + inference_released_gpu_num
                + available_gpu_num
            )

            # Trying to allocate the resource to actor
            increment_gpu_num = await self.actor_manager.allocate_resource(
                total_available_gpu_num, train_iter
            )
            available_gpu_num = total_available_gpu_num - increment_gpu_num
