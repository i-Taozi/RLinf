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

import bisect
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ParallelState:
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    world_size: int = 0
    data_parallel_size: int = 0

    def __post_init__(self):
        self.model_parallel_size = (
            self.tensor_model_parallel_size * self.pipeline_model_parallel_size
        )
        self.valid_dp_sizes = []

    def set_valid_dp_sizes(self, valid_dp_sizes: List[int]):
        self.valid_dp_sizes = valid_dp_sizes

    def allocation(self, available_gpus: int) -> int:
        """Allocate resources to the parallel state.

        Args:
            available_gpus (int): Available GPUs

        Returns:
            Idle GPUs
        """
        if available_gpus < self.model_parallel_size:
            return available_gpus

        incremental_dp_size = available_gpus // self.model_parallel_size

        if len(self.valid_dp_sizes) > 0:
            assert (
                self.data_parallel_size == 0
                or self.data_parallel_size in self.valid_dp_sizes
            ), (
                f"Before allocation, dp_size={self.data_parallel_size} is not in valid_dp_sizes={self.valid_dp_sizes}"
            )
            lower_bound_index = bisect.bisect_left(
                self.valid_dp_sizes, self.data_parallel_size + incremental_dp_size
            )
            if lower_bound_index == len(self.valid_dp_sizes) or (
                self.data_parallel_size + incremental_dp_size
                < self.valid_dp_sizes[lower_bound_index]
            ):
                lower_bound_index -= 1

            valid_dp_size = self.valid_dp_sizes[lower_bound_index]

            incremental_dp_size = valid_dp_size - self.data_parallel_size

        idle_gpus = available_gpus - incremental_dp_size * self.model_parallel_size
        incremental_world_size = incremental_dp_size * self.model_parallel_size

        self.world_size += incremental_world_size
        self.data_parallel_size += incremental_dp_size
        return idle_gpus

    def to_dict(self) -> Dict[str, int]:
        return {
            "world_size": self.world_size,
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }

    def __str__(self) -> str:
        return f"{{world_size={self.world_size}, tensor_model_parallel_size={self.tensor_model_parallel_size}, data_parallel_size={self.data_parallel_size}, pipeline_model_parallel_size={self.pipeline_model_parallel_size}}}"

    def __hash__(self) -> int:
        return hash(
            (
                self.world_size,
                self.tensor_model_parallel_size,
                self.data_parallel_size,
                self.pipeline_model_parallel_size,
            )
        )


class ComponentAllocation:
    def __init__(self, components_config: Dict[str, Dict[str, int]]):
        self.states = {}
        for component_name, component_config in components_config.items():
            self.states[component_name] = ParallelState(
                tensor_model_parallel_size=component_config[
                    "tensor_model_parallel_size"
                ],
                pipeline_model_parallel_size=component_config[
                    "pipeline_model_parallel_size"
                ],
            )
        self.idle_gpus = 0

    def get_component(self, component_name: str) -> ParallelState:
        return self.states.get(component_name)

    def total_gpus(self) -> int:
        assert self.idle_gpus >= 0, (
            f"Idle GPUs must be non-negative, idle_gpus={self.idle_gpus}"
        )
        return self.idle_gpus + self.used_gpus()

    def used_gpus(self) -> int:
        return sum([state.world_size for state in self.states.values()])

    def __str__(self) -> str:
        return f"total_used_gpus={self.used_gpus()}\n{self.states}"


class ResourcePlanner:
    """The Planner for distributing GPU resources to various components of the RL framework"""

    def __init__(
        self,
        components_config: Dict[str, Dict[str, int]],
        total_gpus: int,
        valid_trainer_dp_sizes: List[int],
        valid_inference_dp_sizes: List[int],
    ):
        self.components_config = components_config
        self.initial_allocation = ComponentAllocation(components_config)
        self.total_gpus = total_gpus

        trainer_state = self.initial_allocation.get_component("trainer")
        if trainer_state:
            trainer_state.set_valid_dp_sizes(valid_trainer_dp_sizes)
        inference_state = self.initial_allocation.get_component("inference")
        if inference_state:
            inference_state.set_valid_dp_sizes(valid_inference_dp_sizes)

        self.valid_components = []
        for component in self.components_config.keys():
            self.valid_components.append(component)
        assert len(self.valid_components) in [1, 2, 3]

    def generate_all_states(self) -> List[ComponentAllocation]:
        """Generate all possible resource allocation states during training progression."""

        self.all_states = []

        def trace_recursive(
            current_allocation: ComponentAllocation,
            components: List[str],
        ) -> List[ComponentAllocation]:
            if not components:
                self.all_states.append(current_allocation)
                return

            states = self.generate_states_for_single_component(
                current_allocation, components[0]
            )

            for cur_allocation in states:
                trace_recursive(cur_allocation, components[1:])

        # Generate all states
        trace_recursive(self.initial_allocation, self.valid_components)

    def generate_states_for_single_component(
        self,
        init_allocation: ComponentAllocation,
        component: str,
    ) -> List[ComponentAllocation]:
        avaliable_gpus = self.total_gpus - init_allocation.used_gpus()
        if (
            avaliable_gpus
            < init_allocation.get_component(component).model_parallel_size
        ):
            return []

        min_instance_num = 1
        max_instance_num = (
            avaliable_gpus
            // init_allocation.get_component(component).model_parallel_size
        )

        states = []
        for instance_num in range(min_instance_num, max_instance_num + 1):
            cur_allocation = deepcopy(init_allocation)
            gpu_needed = (
                instance_num
                * init_allocation.get_component(component).model_parallel_size
            )
            if gpu_needed <= avaliable_gpus:
                cur_allocation.get_component(component).allocation(gpu_needed)
                if cur_allocation not in states:
                    states.append(cur_allocation)

        return states

    def generate_static_states(self) -> List[ComponentAllocation]:
        if not hasattr(self, "all_states"):
            self.generate_all_states()

        self.static_states = []
        for allocation_state in self.all_states:
            for component in self.valid_components:
                if allocation_state.get_component(component).data_parallel_size == 0:
                    continue

            if allocation_state.used_gpus() != self.total_gpus:
                continue

            self.static_states.append(allocation_state)

        return self.static_states


def get_valid_dp_sizes(
    total_gpus: int,
    parallel_config: Dict[str, int],
    group_size: int,
    rollout_batch_size: int,
    train_iter: int,
    is_shared: bool,
) -> List[int]:
    global_step_batch_size = rollout_batch_size * group_size
    assert global_step_batch_size % train_iter == 0, (
        f"global_step_batch_size={global_step_batch_size} must be divisible by train_iter={train_iter}"
    )
    trainer_iter_batch_size = global_step_batch_size // train_iter

    valid_dp_sizes = []
    model_parallel_size = (
        parallel_config["tensor_model_parallel_size"]
        * parallel_config["pipeline_model_parallel_size"]
    )
    max_dp_size = total_gpus // model_parallel_size
    get_batch_size = 1 if is_shared else group_size

    for dp_size in range(1, max_dp_size + 1):
        if trainer_iter_batch_size % (dp_size * get_batch_size) == 0:
            valid_dp_sizes.append(dp_size)

    return valid_dp_sizes


def resource_allocate(
    components_config: Dict[str, Dict[str, int]],
    total_gpus: int,
    group_size=8,
    rollout_batch_size=256,
    train_iter=4,
    is_shared=False,
    inference_instance_max_num: int = 2,
) -> List[ComponentAllocation]:
    # Check components config
    for parallel_config in components_config.values():
        assert (
            parallel_config["tensor_model_parallel_size"] > 0
            and parallel_config["pipeline_model_parallel_size"] > 0
        ), (
            "tensor_model_parallel_size and pipeline_model_parallel_size must be greater than 0"
        )

    # Generate valid DP sizes for Inference and Trainer
    valid_inference_dp_sizes: List[int] = []
    valid_trainer_dp_sizes: List[int] = []

    if components_config.get("trainer", None) is not None:
        valid_trainer_dp_sizes: List[int] = get_valid_dp_sizes(
            total_gpus,
            components_config["trainer"],
            group_size,
            rollout_batch_size,
            train_iter,
            is_shared,
        )
    if components_config.get("inference", None) is not None:
        valid_inference_dp_sizes: List[int] = get_valid_dp_sizes(
            total_gpus,
            components_config["inference"],
            group_size,
            rollout_batch_size,
            train_iter,
            is_shared,
        )
        valid_inference_dp_sizes = [
            i for i in valid_inference_dp_sizes if i <= inference_instance_max_num
        ]

    planner = ResourcePlanner(
        components_config=components_config,
        valid_trainer_dp_sizes=valid_trainer_dp_sizes,
        valid_inference_dp_sizes=valid_inference_dp_sizes,
        total_gpus=total_gpus,
    )

    return planner.generate_static_states()
