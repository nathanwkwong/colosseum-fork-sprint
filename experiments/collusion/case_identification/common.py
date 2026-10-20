from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments.collusion.resume import REQUIRED_RUN_FILES, _is_run_complete
from experiments.common.run_utils import load_yaml, write_json


CONTROL_SECRET = False
CONTROL_PROMPT = "control"
COORDINATED_SECRET = True
COORDINATED_PROMPT = "simple"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def case_id(*, seed: int, label: str) -> str:
    return f"seed{int(seed)}__{label}"


def case_identification_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp = cfg.get("experiment") or {}
    block = exp.get("case_identification") or {}
    return dict(block) if isinstance(block, dict) else {}


def results_dir(cfg: Dict[str, Any], *, override: Optional[Path] = None) -> Path:
    if override is not None:
        return Path(override)
    raw = str(case_identification_cfg(cfg).get("results_dir") or "").strip()
    if raw:
        return Path(raw)
    return Path("study_data/results/case_types")


def iter_run_dirs(root: Path) -> List[Path]:
    runs_root = Path(root) / "runs"
    if not runs_root.exists():
        return []
    out: List[Path] = []
    for path in sorted(runs_root.glob("*/*/*")):
        if path.is_dir() and (path / "run_config.json").exists():
            out.append(path)
    return out


def load_run(run_dir: Path) -> Dict[str, Any]:
    run_cfg = read_json(run_dir / "run_config.json")
    summary = read_json(run_dir / "final_summary.json")
    metrics = read_json(run_dir / "metrics.json")
    if not isinstance(run_cfg, dict):
        raise TypeError(f"{run_dir}: run_config.json must be an object")
    if not isinstance(summary, dict):
        raise TypeError(f"{run_dir}: final_summary.json must be an object")
    if not isinstance(metrics, dict):
        raise TypeError(f"{run_dir}: metrics.json must be an object")
    return {
        "run_dir": str(run_dir),
        "run_id": str(run_cfg.get("run_id") or run_dir.name),
        "run_config": run_cfg,
        "final_summary": summary,
        "metrics": metrics,
        "seed": int(run_cfg.get("seed")),
        "secret_channel_enabled": bool(run_cfg.get("secret_channel_enabled", False)),
        "prompt_variant": str(run_cfg.get("prompt_variant") or "").strip() or "control",
        "colluders": [str(a) for a in (run_cfg.get("colluders") or [])],
        "assignment": _assignment_map(summary),
        "agent_rewards": _reward_map(summary),
    }


def _assignment_map(summary: Dict[str, Any]) -> Dict[str, Optional[str]]:
    raw = summary.get("assignment") or {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Optional[str]] = {}
    for k, v in raw.items():
        if v in (None, "skip"):
            out[str(k)] = None
        else:
            out[str(k)] = str(v)
    return out


def _reward_map(summary: Dict[str, Any]) -> Dict[str, float]:
    raw = summary.get("agent_rewards") or {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def is_control_arm(run: Dict[str, Any]) -> bool:
    return (not bool(run.get("secret_channel_enabled"))) and str(
        run.get("prompt_variant") or ""
    ) == CONTROL_PROMPT


def is_coordinated_arm(run: Dict[str, Any]) -> bool:
    return bool(run.get("secret_channel_enabled")) and str(
        run.get("prompt_variant") or ""
    ) == COORDINATED_PROMPT


def complete_run_count(root: Path) -> int:
    n = 0
    for run_dir in iter_run_dirs(root):
        if _is_run_complete(run_dir) and all(
            (run_dir / name).exists() for name in REQUIRED_RUN_FILES
        ):
            n += 1
    return n


def _timestamp_dirs(out_base: Path) -> List[Path]:
    if not out_base.exists():
        return []
    candidates: List[Path] = []
    for child in out_base.iterdir():
        if child.is_dir() and (child / "config.json").exists():
            candidates.append(child)
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates


def find_complete_logs_root(
    cfg: Dict[str, Any],
    *,
    explicit: Optional[Path] = None,
    expected_runs: int = 32,
) -> Optional[Path]:
    if explicit is not None:
        root = Path(explicit)
        if not root.exists():
            raise FileNotFoundError(f"logs root does not exist: {root}")
        return root

    out_base = Path(
        str((cfg.get("experiment") or {}).get("output_dir") or "")
    ).expanduser()
    for root in _timestamp_dirs(out_base):
        if complete_run_count(root) >= int(expected_runs):
            return root
    return None


def find_latest_logs_root(
    cfg: Dict[str, Any],
    *,
    explicit: Optional[Path] = None,
) -> Optional[Path]:
    if explicit is not None:
        root = Path(explicit)
        if not root.exists():
            raise FileNotFoundError(f"logs root does not exist: {root}")
        return root
    out_base = Path(
        str((cfg.get("experiment") or {}).get("output_dir") or "")
    ).expanduser()
    dirs = _timestamp_dirs(out_base)
    return dirs[0] if dirs else None


def expected_generated_runs(cfg: Dict[str, Any]) -> int:
    from experiments.collusion.resume import _iter_expected_run_specs

    return len(list(_iter_expected_run_specs(cfg)))


def mean_or_none(values: Sequence[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def load_experiment_config(path: Path) -> Dict[str, Any]:
    cfg = load_yaml(str(path))
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"empty or invalid config: {path}")
    return cfg


__all__ = [
    "CONTROL_PROMPT",
    "CONTROL_SECRET",
    "COORDINATED_PROMPT",
    "COORDINATED_SECRET",
    "case_id",
    "case_identification_cfg",
    "complete_run_count",
    "expected_generated_runs",
    "find_complete_logs_root",
    "is_control_arm",
    "is_coordinated_arm",
    "iter_run_dirs",
    "load_experiment_config",
    "load_run",
    "mean_or_none",
    "read_json",
    "results_dir",
    "write_json",
]
