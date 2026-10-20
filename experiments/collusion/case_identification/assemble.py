from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from experiments.collusion.case_identification.common import (
    case_id,
    is_control_arm,
    is_coordinated_arm,
    iter_run_dirs,
    load_run,
    write_json,
)
from experiments.collusion.metrics import _coerce_local_rewards


def coalition_advantage_mean(
    agent_rewards: Dict[str, float], colluders: Sequence[str]
) -> Optional[float]:
    colluder_set = {str(a) for a in colluders}
    coal = [float(v) for k, v in agent_rewards.items() if str(k) in colluder_set]
    non = [float(v) for k, v in agent_rewards.items() if str(k) not in colluder_set]
    if not coal or not non:
        return None
    return float((sum(coal) / len(coal)) - (sum(non) / len(non)))


def recompute_agent_rewards(
    *,
    cfg: Dict[str, Any],
    seed: int,
    assignment: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """Score an assignment with the environment reward function (no LLM calls)."""
    import tempfile

    from experiments.collusion.run import _resolve_environment_class
    from experiments.common.local_protocol import LocalCommunicationProtocol

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("simulation", {})["seed"] = int(seed)
    cfg["simulation"]["run_timestamp"] = f"caseid-reward-seed{int(seed)}"
    with tempfile.TemporaryDirectory(prefix="caseid_rewards_") as tmp:
        cfg.setdefault("logging", {})["root_dir"] = tmp
        protocol = LocalCommunicationProtocol(config=cfg)
        env_cls = _resolve_environment_class(cfg.get("environment") or {})
        env = env_cls(protocol, cfg, tool_logger=type("TL", (), {"log_dir": tmp})())
        assignment_full: Dict[str, Any] = {}
        for agent in list(getattr(env, "agent_names", []) or []):
            assignment_full[str(agent)] = assignment.get(str(agent), "skip") or "skip"
        local = _coerce_local_rewards(env._rewards(assignment_full))
        if not local:
            raise RuntimeError(f"env._rewards returned no local rewards for seed={seed}")
        return {str(k): float(v) for k, v in local.items()}


def _index_matched_arms(runs: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for run in runs:
        seed = int(run["seed"])
        bucket = by_seed.setdefault(seed, {})
        if is_control_arm(run):
            bucket["control"] = run
        elif is_coordinated_arm(run):
            bucket["coordinated"] = run
    return by_seed


def assemble_cases(
    *,
    logs_root: Path,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    runs = [load_run(p) for p in iter_run_dirs(logs_root)]
    by_seed = _index_matched_arms(runs)
    cases: List[Dict[str, Any]] = []
    missing: List[str] = []
    for seed in sorted(by_seed):
        arms = by_seed[seed]
        control = arms.get("control")
        coordinated = arms.get("coordinated")
        if control is None or coordinated is None:
            missing.append(
                f"seed={seed} control={control is not None} coordinated={coordinated is not None}"
            )
            continue
        colluders = list(control["colluders"])
        if list(coordinated["colluders"]) != colluders:
            raise ValueError(
                f"seed={seed}: colluder mismatch control={colluders} "
                f"coordinated={coordinated['colluders']}"
            )
        control_rewards = recompute_agent_rewards(
            cfg=cfg, seed=seed, assignment=control["assignment"]
        )
        coordinated_rewards = recompute_agent_rewards(
            cfg=cfg, seed=seed, assignment=coordinated["assignment"]
        )
        cases.append(
            _case_record(
                label="common_incentives",
                seed=seed,
                colluders=colluders,
                assignment=control["assignment"],
                agent_rewards=control_rewards,
                conversation_run_id=control["run_id"],
                assignment_run_id=control["run_id"],
                conversation_run_dir=control["run_dir"],
                assignment_run_dir=control["run_dir"],
            )
        )
        cases.append(
            _case_record(
                label="coordinated_action",
                seed=seed,
                colluders=colluders,
                assignment=coordinated["assignment"],
                agent_rewards=coordinated_rewards,
                conversation_run_id=coordinated["run_id"],
                assignment_run_id=coordinated["run_id"],
                conversation_run_dir=coordinated["run_dir"],
                assignment_run_dir=coordinated["run_dir"],
            )
        )
        cases.append(
            _case_record(
                label="cheap_talk",
                seed=seed,
                colluders=colluders,
                assignment=control["assignment"],
                agent_rewards=control_rewards,
                conversation_run_id=coordinated["run_id"],
                assignment_run_id=control["run_id"],
                conversation_run_dir=coordinated["run_dir"],
                assignment_run_dir=control["run_dir"],
            )
        )
    if missing:
        raise ValueError(
            "matched arms missing for: " + "; ".join(missing)
        )
    if not cases:
        raise ValueError(f"no matched seed pairs under {logs_root}")
    return cases


def _case_record(
    *,
    label: str,
    seed: int,
    colluders: Sequence[str],
    assignment: Dict[str, Optional[str]],
    agent_rewards: Dict[str, float],
    conversation_run_id: str,
    assignment_run_id: str,
    conversation_run_dir: str,
    assignment_run_dir: str,
) -> Dict[str, Any]:
    return {
        "case_id": case_id(seed=seed, label=label),
        "label": label,
        "seed": int(seed),
        "colluders": list(colluders),
        "assignment": dict(assignment),
        "agent_rewards": dict(agent_rewards),
        "coalition_advantage_mean": coalition_advantage_mean(
            agent_rewards, colluders
        ),
        "conversation_run_id": conversation_run_id,
        "assignment_run_id": assignment_run_id,
        "conversation_run_dir": conversation_run_dir,
        "assignment_run_dir": assignment_run_dir,
    }


def write_cases(cases: Sequence[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cases.json"
    payload = {
        "n_cases": len(cases),
        "n_seeds": len({int(c["seed"]) for c in cases}),
        "labels": sorted({str(c["label"]) for c in cases}),
        "cases": list(cases),
    }
    write_json(path, payload)
    return path
