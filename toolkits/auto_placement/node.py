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

import math
from abc import ABC, abstractmethod
from typing import Optional

from util import get_global_config


class ComponentNode(ABC):
    def __init__(self, role: str):
        self.role = role
        self.config = get_global_config()

        self.max_world_size = self.config.components_config[role].max_world_size
        self.model_parallel_size = self.config.components_config[
            role
        ].model_parallel_size
        self.collocated_cost_total = self.config.components_config[
            role
        ].collocated_cost_total
        self.collocated_cost_per_group_batch = (
            self.collocated_cost_total / self.config.rollout_batch_size
        )
        self._gpu_num_to_cost: dict[int, float] = {
            self.max_world_size: self.collocated_cost_per_group_batch
        }
        self._init_profile_data()

    @abstractmethod
    def _init_profile_data(self): ...

    def _validate_gpu_num(self, gpu_num: int) -> bool:
        return gpu_num % self.model_parallel_size == 0

    def profile(self, gpu_num: int) -> Optional[float]:
        return self._gpu_num_to_cost.get(gpu_num, None)

    def __str__(self):
        return self.role

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return hash(self) == hash(other)


class MegatronNode(ComponentNode):
    """The MegatronNode denotes Actor or Inference which backend is Megatron."""

    def __init__(self, role: str, valid_gpu_nums: list[int] = []):
        self.valid_gpu_nums = valid_gpu_nums
        super().__init__(role)

    def _init_profile_data(self):
        self._hyper_params = {
            "max_scale": 8,
            "bias": 0.5,
        }

        for gpu_num in range(1, self.max_world_size):
            if not self._validate_gpu_num(gpu_num):
                continue
            self._gpu_num_to_cost[gpu_num] = self._estimate_cost(gpu_num)

    def _validate_gpu_num(self, gpu_num: int) -> bool:
        if not self.valid_gpu_nums:
            return super()._validate_gpu_num(gpu_num)
        return gpu_num in self.valid_gpu_nums

    def _estimate_cost(self, gpu_num: int) -> float:
        """Performance model for MegatronNode.

        1. estimated_cost_linear = self.collocated_cost_per_group_batch * scale

        2. scale_ratio = math.log(min(scale, max_scale), 2) + bias.

        3. estimated_cost = estimated_cost_linear / scale_ratio
        """
        scale = self.max_world_size / gpu_num
        estimated_cost_linear = self.collocated_cost_per_group_batch * scale
        scale_ratio = (
            math.log(min(scale, self._hyper_params["max_scale"]), 2)
            + self._hyper_params["bias"]
        )
        estimated_cost = estimated_cost_linear / scale_ratio
        return estimated_cost


class RolloutNode(ComponentNode):
    def __init__(
        self,
    ):
        super().__init__("rollout")

    def _init_profile_data(self):
        queue_wait_ratio = 0.9 if self.config.max_running_requests >= 128 else 1
        self._hyper_params = {
            "max_running_requests_wo_queue_wait": math.ceil(
                self.config.max_running_requests * queue_wait_ratio
            )
        }

        for gpu_num in range(1, self.max_world_size):
            if not self._validate_gpu_num(gpu_num):
                continue
            instance_num = gpu_num // self.model_parallel_size
            if (
                instance_num * self._hyper_params["max_running_requests_wo_queue_wait"]
                >= self.config.rollout_batch_size * self.config.group_size
            ):
                self._gpu_num_to_cost[gpu_num] = self.collocated_cost_per_group_batch
            else:
                self._gpu_num_to_cost[gpu_num] = (
                    self.collocated_cost_per_group_batch
                    * (self.max_world_size / gpu_num)
                )


class EnvNode(ComponentNode):
    def __init__(self):
        self.role = "env"

    def _init_profile_data(self): ...

    def profile(self, gpu_num: int) -> Optional[float]:
        raise NotImplementedError("EnvNode is not implemented")


class SccNode(ComponentNode):
    """The SccNode denotes a strongly connected component (SCC) Node.

    Assert nodes is sorted by execute order.
    """

    def __init__(self, nodes: list[ComponentNode]):
        self.nodes = nodes
        self.role = " - ".join([node.role for node in nodes])

    def _init_profile_data(self): ...

    def profile(self, gpu_num: int) -> Optional[float]:
        raise NotImplementedError("EnvNode is not implemented")
