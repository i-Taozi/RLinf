from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ReshardingState(Enum):
    """Resharding state for the resharding process.
    """
    DEFAULT =       0
    RESHARDING =    1
    RUN =           2
    CONTINUE =      3

@dataclass
class ScheduleReq:
    @staticmethod
    def deserialize(the_dict: dict) -> 'ScheduleReq':
        assert isinstance(the_dict, dict) and 'cls' in the_dict, f"data dict error: {the_dict}"
        dict_cls = the_dict['cls']
        dict_no_cls = {k: v for k, v in the_dict.items() if k != 'cls'}
        cls = _ScheduleReqClasses[dict_cls]
        return cls(**dict_no_cls)

    def serialize(self):
        return {'cls': self.__class__.__name__, **self.__dict__}

@dataclass
class RolloutReq(ScheduleReq):
    type: Literal['offloaded', 'migrated_in', 'migrated_out', 'report']
    rank: int =                 None

    # for migrated_out
    input_ids: list =           None
    results: list =             None
    abort_results: list =       None
    answers: list =             None
    # for migrated_out end

    # for report
    total_requests: int =       None
    completed_requests: int =   None
    total_tasks: int =          None
    completed_tasks: int =      None
    running_tasks: int =        None
    timestamp: float =          None
    # for report end

    def __repr__(self) -> str:
        if self.type == 'migrated_out':
            return f'RolloutReq(type={self.type}, lengths={len(self.input_ids), len(self.results), len(self.abort_results), len(self.answers)})'
        else:
            return f'RolloutReq({", ".join(f"{k}={v}" for k, v in self.__dict__.items())})'

@dataclass
class InferenceReq(ScheduleReq):
    pass

@dataclass
class TrainerReq(ScheduleReq):
    type: Literal['req_loop', 'req_end', 'resharded', 'loop_finish']
    step_id: int = 0
    input_qsize: int = 0
    current_batch_size: int = 0
    micro_batch_counter: int = 0

_ScheduleReqClasses = {
    'RolloutReq': RolloutReq,
    'InferenceReq': InferenceReq,
    'TrainerReq': TrainerReq,
}

@dataclass
class ScheduleResp:
    @classmethod
    def deserialize(cls, the_dict: dict):
        return cls(**the_dict)

    def serialize(self):
        return self.__dict__

@dataclass
class RolloutResp(ScheduleResp):
    type: Literal['offload', 'migrate_in', 'migrate_out']

    # for migrate_out
    # group_name: str = None
    # from_rank: int = None
    # for migrate_out end

    # for migrate_in
    input_ids: list =           None
    results: list =             None
    abort_results: list =       None
    answers: list =             None
    # for migrate_in end

    def __repr__(self) -> str:
        if self.type == 'migrate_in':
            return f'RolloutResp(type={self.type}, lengths={len(self.input_ids), len(self.results), len(self.abort_results), len(self.answers)})'
        else:
            return f'RolloutResp({", ".join(f"{k}={v}" for k, v in self.__dict__.items())})'

@dataclass
class TrainerResp(ScheduleResp):
    need_reshard: bool
    # rank_inner_to_outer: dict[int, int] = None
    tag: str =  None
    ws: int =   None
    # tp: int =   None
    # cp: int =   None
    # tep: int =  None
    # ep: int =   None
    # dp: int =   None
    # pp: int =   None
