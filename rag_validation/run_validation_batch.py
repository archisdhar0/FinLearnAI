"""
Run rag_validation/validate_rag.py multiple times and summarize variability.

Why this exists:
  - validate_rag.py overwrites rag_validation/validation_results.json each run.
  - LLM answers can vary run-to-run, so we want a distribution of outcomes.

Outputs (created under rag_validation/validation_runs/):
  - results_run_XXX.json: full per-question results for each run
  - run_summaries.jsonl: one JSON line per run with aggregate metrics
  - summary.json / summary.csv: aggregate stats across runs

Usage:
  cd FinLearnAI
  python3 rag_validation/run_validation_batch.py --runs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RunSummary:
    run_index: int
    exit_code: int
    total_questions: int
    passed_questions: int
    pass_rate: float
    mean_similarity: Optional[float]
    min_similarity: Optional[float]
    max_similarity: Optional[float]
    mean_key_fact_coverage: float
    seconds: float
    results_file: str


def _safe_mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _basic_stats(xs: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not xs:
        return None, None, None, None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return mean, math.sqrt(var), min(xs), max(xs)


def _ascii_hist(values: List[float], *, bins: int = 10, lo: float = 0.0, hi: float = 1.0, width: int = 28) -> str:
    if not values:
        return "(no data)"
    if hi <= lo:
        return "(invalid range)"

    counts = [0] * bins
    for v in values:
        if v is None:
            continue
        v_clamped = min(max(v, lo), hi)
        idx = int((v_clamped - lo) / (hi - lo) * bins)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1

    max_c = max(counts) if counts else 1
    lines = []
    for i, c in enumerate(counts):
        b_lo = lo + (hi - lo) * (i / bins)
        b_hi = lo + (hi - lo) * ((i + 1) / bins)
        bar_len = int(round((c / max_c) * width)) if max_c else 0
        bar = "#" * bar_len
        lines.append(f"{b_lo:0.2f}–{b_hi:0.2f} | {bar} {c}")
    return "\n".join(lines)


def _load_validation_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_results(results: List[Dict[str, Any]], *, run_index: int, exit_code: int, seconds: float, results_file: str) -> RunSummary:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed") is True)
    pass_rate = (passed / total) if total else 0.0

    sims = [r["similarity"] for r in results if isinstance(r.get("similarity"), (int, float))]
    mean_sim = _safe_mean([float(x) for x in sims]) if sims else None
    min_sim = min(sims) if sims else None
    max_sim = max(sims) if sims else None

    coverages = [r.get("key_fact_coverage") for r in results]
    coverage_vals = [float(c) for c in coverages if isinstance(c, (int, float))]
    mean_cov = (sum(coverage_vals) / len(coverage_vals)) if coverage_vals else 0.0

    return RunSummary(
        run_index=run_index,
        exit_code=exit_code,
        total_questions=total,
        passed_questions=passed,
        pass_rate=pass_rate,
        mean_similarity=float(mean_sim) if mean_sim is not None else None,
        min_similarity=float(min_sim) if min_sim is not None else None,
        max_similarity=float(max_sim) if max_sim is not None else None,
        mean_key_fact_coverage=mean_cov,
        seconds=seconds,
        results_file=results_file,
    )


def run_once(repo_root: Path, *, run_index: int, python: str, timeout_s: Optional[int] = None) -> Tuple[int, str, float]:
    """
    Run validate_rag.py once.
    Returns: (exit_code, stdout_text, seconds_elapsed)
    """
    cmd = [python, "rag_validation/validate_rag.py"]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=os.environ.copy(),
    )
    dt = time.time() - t0
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30, help="Number of validation runs to execute.")
    ap.add_argument("--python", default=sys.executable or "python3", help="Python executable to use.")
    ap.add_argument("--timeout-s", type=int, default=None, help="Timeout per run (seconds).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "rag_validation" / "validation_results.json"
    out_dir = repo_root / "rag_validation" / "validation_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "run_summaries.jsonl"
    # Start fresh each batch so the file matches this invocation.
    if jsonl_path.exists():
        jsonl_path.unlink()

    summaries: List[RunSummary] = []

    for i in range(1, args.runs + 1):
        print(f"\n=== Run {i}/{args.runs} ===")
        code, output, seconds = run_once(repo_root, run_index=i, python=args.python, timeout_s=args.timeout_s)
        print(output.strip())

        if not results_path.exists():
            print(f"\nERROR: Expected results file missing: {results_path}")
            return 2

        results = _load_validation_results(results_path)
        run_file = out_dir / f"results_run_{i:03d}.json"
        shutil.copyfile(results_path, run_file)

        summary = _summarize_results(
            results,
            run_index=i,
            exit_code=code,
            seconds=seconds,
            results_file=str(run_file.relative_to(repo_root)),
        )
        summaries.append(summary)

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(summary), ensure_ascii=False) + "\n")

        print(
            f"Run {i}: pass_rate={summary.pass_rate:.3f} "
            f"mean_sim={summary.mean_similarity if summary.mean_similarity is not None else 'n/a'} "
            f"mean_cov={summary.mean_key_fact_coverage:.3f} "
            f"({summary.seconds:.1f}s)"
        )

    # Aggregate summary
    pass_rates = [s.pass_rate for s in summaries]
    mean_sims = [s.mean_similarity for s in summaries if s.mean_similarity is not None]
    mean_covs = [s.mean_key_fact_coverage for s in summaries]

    pr_mean, pr_std, pr_min, pr_max = _basic_stats(pass_rates)
    ms_mean, ms_std, ms_min, ms_max = _basic_stats([float(x) for x in mean_sims]) if mean_sims else (None, None, None, None)
    mc_mean, mc_std, mc_min, mc_max = _basic_stats(mean_covs)

    summary_obj = {
        "runs": len(summaries),
        "pass_rate": {"mean": pr_mean, "std": pr_std, "min": pr_min, "max": pr_max},
        "mean_similarity": {"mean": ms_mean, "std": ms_std, "min": ms_min, "max": ms_max},
        "mean_key_fact_coverage": {"mean": mc_mean, "std": mc_std, "min": mc_min, "max": mc_max},
        "outputs": {
            "run_summaries_jsonl": str(jsonl_path.relative_to(repo_root)),
            "results_dir": str(out_dir.relative_to(repo_root)),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    # CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "exit_code",
                "total_questions",
                "passed_questions",
                "pass_rate",
                "mean_similarity",
                "min_similarity",
                "max_similarity",
                "mean_key_fact_coverage",
                "seconds",
                "results_file",
            ],
        )
        w.writeheader()
        for s in summaries:
            w.writerow(asdict(s))

    print("\n=== Distribution: pass_rate ===")
    print(_ascii_hist(pass_rates, bins=10, lo=0.0, hi=1.0))
    if mean_sims:
        print("\n=== Distribution: mean_similarity ===")
        print(_ascii_hist([float(x) for x in mean_sims], bins=10, lo=0.0, hi=1.0))
    print("\nWrote:")
    print(f"  - {out_dir / 'summary.json'}")
    print(f"  - {csv_path}")
    print(f"  - {jsonl_path}")
    print(f"  - {out_dir / 'results_run_001.json'} ...")

    # Non-zero if any run had a failure (exit_code != 0) OR pass_rate < 1.0
    any_bad = any(s.exit_code != 0 or s.passed_questions != s.total_questions for s in summaries)
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

