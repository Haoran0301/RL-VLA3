from dataclasses import dataclass
from typing import Dict, List, Optional

from omegaconf import DictConfig


@dataclass
class SlotSpec:
    slot_id: int  # global unique id for internal indexing (buffers, queues)
    local_slot_index: int  # 0,1,2,... within this global_env_id (for channel key)
    env_rank: int
    global_env_id: int
    local_manager_idx: int
    rollout_rank: int
    batch_size: int


@dataclass
class FlexPlan:
    enabled: bool
    manager_batch_sizes_by_env_rank: List[List[int]]
    global_env_id_by_env_rank: List[List[int]]
    slot_specs_by_global_env_id: Dict[int, List[SlotSpec]]
    slot_specs_by_rollout_rank: Dict[int, List[SlotSpec]]
    total_slot_count: int


def _get_nested(cfg: DictConfig, path: str, default):
    cur = cfg
    for part in path.split("."):
        if cur is None or not hasattr(cur, "get"):
            return default
        cur = cur.get(part, None)
        if cur is None:
            return default
    return cur


def _normalize_manager_batch_sizes(
    cfg: DictConfig,
    env_world_size: int,
    default_stage_num: int,
    default_batch_size: int,
) -> List[List[int]]:
    flex_cfg = _get_nested(cfg, "env.per_env_async.flex", None)
    custom = None if flex_cfg is None else flex_cfg.get("manager_batch_sizes_by_env_rank", None)
    if custom is None:
        return [[int(default_batch_size) for _ in range(default_stage_num)] for _ in range(env_world_size)]  # Use defaults computed from total_num_envs and pipeline settings.

    custom = list(custom)
    if len(custom) != env_world_size:
        raise ValueError(
            "env.per_env_async.flex.manager_batch_sizes_by_env_rank length must equal env_world_size. "
            f"Got {len(custom)} vs {env_world_size}"
        )
    normalized: List[List[int]] = []
    for env_rank, per_rank in enumerate(custom):
        arr = [int(x) for x in list(per_rank)]
        if len(arr) == 0:
            raise ValueError(f"env rank {env_rank} manager list must not be empty")
        if any(x <= 0 for x in arr):
            raise ValueError(f"env rank {env_rank} manager batch sizes must be > 0")
        normalized.append(arr)

    print(f"[DEBUG _normalize_manager_batch_sizes] normalized: {normalized}")
    return normalized


def _normalize_rollout_ranks(
    cfg: DictConfig,
    manager_batch_sizes_by_env_rank: List[List[int]],
    rollout_world_size: int,
) -> List[List[List[int]]]:
    flex_cfg = _get_nested(cfg, "rollout.per_env_async.flex", None)
    custom = None if flex_cfg is None else flex_cfg.get("rollout_ranks_by_env_rank_stage", None)
    # [DEBUG_FLEX_PLAN]
    print(f"[DEBUG _normalize_rollout_ranks] flex_cfg={flex_cfg is not None}, custom={custom is not None}")
    if custom is None:
        all_rollouts = list(range(rollout_world_size))
        return [[list(all_rollouts) for _ in per_rank] for per_rank in manager_batch_sizes_by_env_rank]

    custom = list(custom)
    if len(custom) != len(manager_batch_sizes_by_env_rank):
        raise ValueError(
            "rollout.per_env_async.flex.rollout_ranks_by_env_rank_stage env dimension mismatch: "
            f"{len(custom)} vs {len(manager_batch_sizes_by_env_rank)}"
        )

    normalized: List[List[List[int]]] = []
    for env_rank, (per_rank_custom, per_rank_mgr) in enumerate(zip(custom, manager_batch_sizes_by_env_rank)):
        per_rank_custom = list(per_rank_custom)
        if len(per_rank_custom) != len(per_rank_mgr):
            raise ValueError(
                "rollout ranks stage dimension mismatch for env rank "
                f"{env_rank}: {len(per_rank_custom)} vs {len(per_rank_mgr)}"
            )
        per_rank_norm: List[List[int]] = []
        for stage_idx, rollout_ranks in enumerate(per_rank_custom):
            rr = [int(x) for x in list(rollout_ranks)]
            if len(rr) == 0:
                raise ValueError(
                    f"rollout ranks for env_rank={env_rank}, stage={stage_idx} must not be empty"
                )
            for r in rr:
                if r < 0 or r >= rollout_world_size:
                    raise ValueError(
                        f"invalid rollout rank {r}; valid range [0, {rollout_world_size - 1}]"
                    )
            per_rank_norm.append(rr)
        normalized.append(per_rank_norm)
    return normalized


def _normalize_split_sizes(
    cfg: DictConfig,
    manager_batch_sizes_by_env_rank: List[List[int]],
    rollout_ranks_by_env_rank_stage: List[List[List[int]]],
) -> List[List[Optional[List[int]]]]:
    flex_cfg = _get_nested(cfg, "rollout.per_env_async.flex", None)
    custom = None if flex_cfg is None else flex_cfg.get("split_sizes_by_env_rank_stage", None)
    if custom is None:
        return [[None for _ in per_rank] for per_rank in manager_batch_sizes_by_env_rank]

    custom = list(custom)
    if len(custom) != len(manager_batch_sizes_by_env_rank):
        raise ValueError(
            "split sizes env dimension mismatch: "
            f"{len(custom)} vs {len(manager_batch_sizes_by_env_rank)}"
        )
    normalized: List[List[Optional[List[int]]]] = []
    for env_rank, (per_rank_custom, per_rank_mgr, per_rank_rr) in enumerate(
        zip(custom, manager_batch_sizes_by_env_rank, rollout_ranks_by_env_rank_stage)
    ):
        per_rank_custom = list(per_rank_custom)
        if len(per_rank_custom) != len(per_rank_mgr):
            raise ValueError(
                "split sizes stage dimension mismatch for env rank "
                f"{env_rank}: {len(per_rank_custom)} vs {len(per_rank_mgr)}"
            )
        row: List[Optional[List[int]]] = []
        for stage_idx, value in enumerate(per_rank_custom):
            if value is None:
                row.append(None)
                continue
            arr = [int(x) for x in list(value)]
            if len(arr) != len(per_rank_rr[stage_idx]):
                raise ValueError(
                    "split size length must equal rollout ranks length for "
                    f"env_rank={env_rank}, stage={stage_idx}"
                )
            if any(x <= 0 for x in arr):
                raise ValueError("split sizes must be > 0")
            if sum(arr) != int(per_rank_mgr[stage_idx]):
                raise ValueError(
                    "split sizes sum mismatch for "
                    f"env_rank={env_rank}, stage={stage_idx}: sum={sum(arr)} "
                    f"vs manager_batch={int(per_rank_mgr[stage_idx])}"
                )
            row.append(arr)
        normalized.append(row)
    return normalized


def build_flex_plan(
    cfg: DictConfig,
    env_world_size: int,
    rollout_world_size: int,
    default_stage_num: int,
    default_batch_size: int,
) -> FlexPlan:
    env_flex_cfg = _get_nested(cfg, "env.per_env_async.flex", None)
    rollout_flex_cfg = _get_nested(cfg, "rollout.per_env_async.flex", None)
    enabled = bool(
        (env_flex_cfg is not None and env_flex_cfg.get("enabled", False))
        or (rollout_flex_cfg is not None and rollout_flex_cfg.get("enabled", False))
    )

    manager_batch_sizes_by_env_rank = _normalize_manager_batch_sizes(
        cfg=cfg,
        env_world_size=env_world_size,
        default_stage_num=default_stage_num,
        default_batch_size=default_batch_size,
    )
    rollout_ranks_by_env_rank_stage = _normalize_rollout_ranks(
        cfg=cfg,
        manager_batch_sizes_by_env_rank=manager_batch_sizes_by_env_rank,
        rollout_world_size=rollout_world_size,
    )
    split_sizes_by_env_rank_stage = _normalize_split_sizes(
        cfg=cfg,
        manager_batch_sizes_by_env_rank=manager_batch_sizes_by_env_rank,
        rollout_ranks_by_env_rank_stage=rollout_ranks_by_env_rank_stage,
    )

    # Allocate global_env_ids by manager index first (not by env_rank)
    # This ensures rollout workers process all env_rank's manager 0, then manager 1, etc.
    # For 5 env_ranks × 3 managers: [0,5,10], [1,6,11], [2,7,12], [3,8,13], [4,9,14]
    max_managers = max(len(per_rank) for per_rank in manager_batch_sizes_by_env_rank)
    global_env_id_by_env_rank: List[List[int]] = [
        [] for _ in range(env_world_size)
    ]
    next_env_id = 0
    for manager_idx in range(max_managers):
        for env_rank in range(env_world_size):
            if manager_idx < len(manager_batch_sizes_by_env_rank[env_rank]):
                global_env_id_by_env_rank[env_rank].append(next_env_id)
                next_env_id += 1

    slot_specs_by_global_env_id: Dict[int, List[SlotSpec]] = {}
    slot_specs_by_rollout_rank: Dict[int, List[SlotSpec]] = {
        r: [] for r in range(rollout_world_size)
    }
    next_slot_id = 0
    for env_rank, (mgr_sizes, mgr_env_ids, mgr_rollout_ranks, mgr_split_sizes) in enumerate(
        zip(
            manager_batch_sizes_by_env_rank,
            global_env_id_by_env_rank,
            rollout_ranks_by_env_rank_stage,
            split_sizes_by_env_rank_stage,
        )
    ):
        for local_manager_idx, (manager_batch_size, global_env_id, rollout_ranks, split_sizes) in enumerate(
            zip(mgr_sizes, mgr_env_ids, mgr_rollout_ranks, mgr_split_sizes)
        ):
            n_slots = len(rollout_ranks)
            if split_sizes is None:
                base = int(manager_batch_size) // n_slots
                rem = int(manager_batch_size) % n_slots
                split_sizes = [base + (1 if i < rem else 0) for i in range(n_slots)]
                if any(x == 0 for x in split_sizes):
                    raise ValueError(
                        "manager batch is too small for rollout split count: "
                        f"manager_batch={manager_batch_size}, n_slots={n_slots}"
                    )
            slot_specs: List[SlotSpec] = []
            for local_slot_index, (rollout_rank, slot_batch_size) in enumerate(
                zip(rollout_ranks, split_sizes)
            ):
                spec = SlotSpec(
                    slot_id=next_slot_id,
                    local_slot_index=local_slot_index,
                    env_rank=env_rank,
                    global_env_id=global_env_id,
                    local_manager_idx=local_manager_idx,
                    rollout_rank=int(rollout_rank),
                    batch_size=int(slot_batch_size),
                )
                slot_specs.append(spec)
                slot_specs_by_rollout_rank[int(rollout_rank)].append(spec)
                next_slot_id += 1
            slot_specs_by_global_env_id[global_env_id] = slot_specs

    return FlexPlan(
        enabled=enabled,
        manager_batch_sizes_by_env_rank=manager_batch_sizes_by_env_rank,
        global_env_id_by_env_rank=global_env_id_by_env_rank,
        slot_specs_by_global_env_id=slot_specs_by_global_env_id,
        slot_specs_by_rollout_rank=slot_specs_by_rollout_rank,
        total_slot_count=next_slot_id,
    )
