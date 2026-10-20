from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.collusion.case_identification.assemble import assemble_cases, write_cases
from experiments.collusion.case_identification.common import (
    complete_run_count,
    expected_generated_runs,
    find_complete_logs_root,
    find_latest_logs_root,
    load_experiment_config,
    results_dir,
    write_json,
)
from experiments.collusion.case_identification.freeze import freeze_and_evaluate
from experiments.collusion.case_identification.score import score_cases
from experiments.collusion.resume import resume_runs
from experiments.collusion.run import run_from_config


def _write_summary_md(
    *,
    out_dir: Path,
    logs_root: Path,
    n_cases: int,
    report: Dict[str, Any],
    frozen: Dict[str, Any],
) -> None:
    check = frozen.get("manipulation_check") or {}
    means = check.get("means") or {}
    thresh = (report.get("test") or {}).get("threshold") or {}
    logistic = (report.get("test") or {}).get("logistic") or {}
    lines = [
        "# Three-case identification report",
        "",
        f"- Logs: `{logs_root}`",
        f"- Cases: {n_cases}",
        f"- Train seeds: {frozen.get('train_seeds')}",
        f"- Test seeds: {frozen.get('test_seeds')}",
        f"- Manipulation failed: {bool(report.get('manipulation_failed'))}",
    ]
    if report.get("manipulation_reason"):
        lines.append(f"- Manipulation note: {report['manipulation_reason']}")
    lines.extend(["", "## Training means", ""])
    for label, stats in means.items():
        lines.append(
            f"- `{label}`: conversation={stats.get('conversation')}, "
            f"advantage_residual={stats.get('coalition_advantage_residual')}, "
            f"dependence={stats.get('dependence')}"
        )
    rule = frozen.get("threshold_rule") or {}
    lines.extend(
        [
            "",
            "## Frozen threshold rule",
            "",
            f"- d* (dependence) = {rule.get('d_star')}",
            f"- a* (advantage residual) = {rule.get('a_star')}",
            f"- c* (conversation) = {rule.get('c_star')}",
            f"- Train macro-F1 = {rule.get('train_macro_f1')}",
            "",
            "## Held-out test",
            "",
            f"- Threshold macro-F1 = {thresh.get('macro_f1')}",
            f"- Logistic macro-F1 = {logistic.get('macro_f1')}",
            f"- Common incentives called coordinated (threshold) = "
            f"{(thresh.get('errors_of_interest') or {}).get('common_incentives_called_coordinated')}",
            f"- Cheap talk called coordinated (threshold) = "
            f"{(thresh.get('errors_of_interest') or {}).get('cheap_talk_called_coordinated')}",
            "",
            "Threshold confusion (true × pred):",
            "",
            "```json",
            json.dumps(thresh.get("confusion") or {}, indent=2),
            "```",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


async def _ensure_logs(
    *,
    cfg: Dict[str, Any],
    config_path: Path,
    logs_root: Optional[Path],
    skip_generate: bool,
) -> Path:
    expected = expected_generated_runs(cfg)
    complete = find_complete_logs_root(
        cfg, explicit=logs_root, expected_runs=expected
    )
    if complete is not None and complete_run_count(complete) >= expected:
        return complete
    latest = find_latest_logs_root(cfg, explicit=logs_root)
    if skip_generate:
        if latest is None:
            raise FileNotFoundError(
                "No complete logs root found and --skip-generate was set."
            )
        raise FileNotFoundError(
            f"Logs at {latest} have {complete_run_count(latest)}/{expected} complete runs."
        )

    if latest is not None:
        await resume_runs(
            root=latest,
            cfg=cfg,
            max_concurrent_runs=int(
                (cfg.get("experiment") or {}).get("max_concurrent_runs", 1)
            ),
            stop_on_error=True,
            require_status_complete=True,
            rerun_error_turns=True,
            dry_run=False,
        )
        return latest

    generated = await run_from_config(str(config_path))
    return Path(generated)


async def run_pipeline(
    *,
    config_path: Path,
    logs_root: Optional[Path],
    results_override: Optional[Path],
    skip_generate: bool,
    skip_judge: bool,
    max_concurrent_judge: int,
) -> Path:
    cfg = load_experiment_config(config_path)
    out_dir = results_dir(cfg, override=results_override)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = await _ensure_logs(
        cfg=cfg,
        config_path=config_path,
        logs_root=logs_root,
        skip_generate=skip_generate,
    )
    write_json(
        out_dir / "run_meta.json",
        {
            "config_path": str(config_path),
            "logs_root": str(root),
            "started_at": datetime.now().isoformat(),
            "expected_generated_runs": expected_generated_runs(cfg),
            "complete_generated_runs": complete_run_count(root),
        },
    )
    cases = assemble_cases(logs_root=root, cfg=cfg)
    write_cases(cases, out_dir)

    scored_path = out_dir / "scored_cases.json"
    if skip_judge and scored_path.exists():
        scored_payload = json.loads(scored_path.read_text(encoding="utf-8"))
        scored = list(scored_payload.get("cases") or [])
    else:
        scored = await score_cases(
            cases=cases,
            cfg=cfg,
            out_dir=out_dir,
            max_concurrent=max_concurrent_judge,
        )
    result = freeze_and_evaluate(scored, cfg=cfg, out_dir=out_dir)
    _write_summary_md(
        out_dir=out_dir,
        logs_root=root,
        n_cases=len(scored),
        report=result["report"],
        frozen=result["frozen"],
    )
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble three matched collusion cases, score shared evidence, "
            "and freeze a decision rule on half the seeds."
        )
    )
    parser.add_argument(
        "--config",
        default="study_data/collusion_case_types_ollama.yaml",
        help="Sweep YAML (default: study_data/collusion_case_types_ollama.yaml).",
    )
    parser.add_argument(
        "--logs-root",
        default=None,
        help="Organiser-supplied or previously generated timestamp folder.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override study_data/results/case_types.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Do not launch missing trajectory generation.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Reuse scored_cases.json if present (no new judge calls).",
    )
    parser.add_argument(
        "--max-concurrent-judge",
        type=int,
        default=4,
        help="Concurrent judge calls (default: 4).",
    )
    args = parser.parse_args(argv)
    out = asyncio.run(
        run_pipeline(
            config_path=Path(args.config),
            logs_root=Path(args.logs_root) if args.logs_root else None,
            results_override=Path(args.results_dir) if args.results_dir else None,
            skip_generate=bool(args.skip_generate),
            skip_judge=bool(args.skip_judge),
            max_concurrent_judge=int(args.max_concurrent_judge),
        )
    )
    print(f"Wrote case-identification results to: {out}")


if __name__ == "__main__":
    main()
