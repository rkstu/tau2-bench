"""
Phase 6: Experiment Runner.

CLI entry point for tau2-TRACE. Supports two modes:

1. Post-hoc analysis: Analyze existing simulation Results JSON files.
2. Live evaluation: Run simulations with optional adversarial wrapper.

Usage:
    # Analyze existing results
    python -m experiments.tau2_trace.run_experiment analyze \
        --results-file data/tau2/simulations/my_run.json \
        --output src/experiments/tau2_trace/results/augmented.csv

    # Analyze with explicit domain override
    python -m experiments.tau2_trace.run_experiment analyze \
        --results-file data/tau2/simulations/my_run.json \
        --domain telecom \
        --output src/experiments/tau2_trace/results/augmented.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from tau2.data_model.simulation import Results

from experiments.tau2_trace.domain_router import evaluate_results_trace


def analyze_results(
    results_path: Path,
    output_path: Path,
    domain_override: str | None = None,
) -> pd.DataFrame:
    """
    Post-hoc analysis mode: load an existing Results file, run tau2-TRACE
    analysis, and produce an augmented DataFrame.

    Args:
        results_path: Path to a tau2-bench simulation Results JSON file.
        output_path: Path to write the augmented CSV.
        domain_override: Override the domain from the Results file.

    Returns:
        The augmented pandas DataFrame.
    """
    logger.info(f"Loading results from {results_path}")
    results = Results.load(results_path)

    domain = domain_override or results.info.environment_info.domain_name
    logger.info(f"Domain: {domain}")
    logger.info(f"Simulations: {len(results.simulations)}")
    logger.info(f"Tasks: {len(results.tasks)}")

    # Build task_id -> expected_actions mapping from task metadata
    task_expected_actions: dict[str, int] = {}
    for task in results.tasks:
        if task.evaluation_criteria is not None:
            info = task.evaluation_criteria.info()
            num_actions = info.get("num_agent_actions", 0) + info.get(
                "num_user_actions", 0
            )
            task_expected_actions[task.id] = num_actions

    # Run tau2-TRACE analysis
    logger.info("Running tau2-TRACE analysis...")
    scorecards = evaluate_results_trace(
        simulations=results.simulations,
        domain=domain,
        task_expected_actions=task_expected_actions,
    )

    # Build the core DataFrame from Results.
    # to_df() may fail if tasks lack evaluation_criteria, so fall back to
    # building a minimal DataFrame from simulation fields.
    try:
        df_core = results.to_df()
    except (KeyError, AttributeError):
        logger.warning(
            "Results.to_df() failed (likely missing evaluation_criteria). "
            "Building minimal DataFrame from simulation data."
        )
        rows = []
        for sim in results.simulations:
            rows.append(
                {
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "seed": sim.seed,
                    "reward": sim.reward_info.reward if sim.reward_info else None,
                    "agent_cost": sim.agent_cost,
                    "user_cost": sim.user_cost,
                    "termination_reason": sim.termination_reason.value,
                    "duration": sim.duration,
                    "num_messages": len(sim.messages),
                    "info_domain": domain,
                }
            )
        df_core = pd.DataFrame(rows)

    # Build the trace DataFrame from scorecards
    trace_rows = [sc.to_dict() for sc in scorecards]
    df_trace = pd.DataFrame(trace_rows)

    # Merge on (task_id, trial)
    merge_cols = ["task_id"]
    if "trial" in df_core.columns and "trial" in df_trace.columns:
        merge_cols.append("trial")

    df_augmented = pd.merge(df_core, df_trace, on=merge_cols, how="left")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_augmented.to_csv(output_path, index=False)
    logger.info(f"Augmented results saved to {output_path}")

    # Print summary
    _print_summary(df_augmented, domain)

    return df_augmented


def _print_summary(df: pd.DataFrame, domain: str) -> None:
    """Print a concise summary of tau2-TRACE metrics."""
    print("\n" + "=" * 70)
    print(f"  tau2-TRACE Analysis Summary  |  Domain: {domain}")
    print("=" * 70)

    trace_cols = [c for c in df.columns if c.startswith("trace_")]
    if not trace_cols:
        print("  No trace metrics computed.")
        return

    # Core outcome metrics
    if "reward" in df.columns:
        avg_reward = df["reward"].mean()
        pass_rate = (df["reward"] >= 1.0 - 1e-6).mean() * 100
        print(f"\n  Core Metrics:")
        print(f"    Average Reward:        {avg_reward:.4f}")
        print(f"    Pass Rate:             {pass_rate:.1f}%")

    # Trajectory efficiency
    print(f"\n  Trajectory Efficiency:")
    for col in [
        "trace_total_turns",
        "trace_total_tool_calls",
        "trace_redundant_tool_calls",
        "trace_loop_count",
        "trace_error_count",
        "trace_error_recovery_rate",
    ]:
        if col in df.columns:
            label = col.replace("trace_", "").replace("_", " ").title()
            print(f"    {label:30s} {df[col].mean():>8.2f} (avg)")

    # Ordering metrics
    print(f"\n  Policy Adherence:")
    for col in [
        "trace_policy_adherence",
        "trace_read_after_write_score",
        "trace_policy_breach_count",
    ]:
        if col in df.columns:
            label = col.replace("trace_", "").replace("_", " ").title()
            print(f"    {label:30s} {df[col].mean():>8.2f} (avg)")

    # Interaction quality
    print(f"\n  Interaction Quality:")
    for col in [
        "trace_action_density",
        "trace_token_to_action_ratio",
        "trace_turns_vs_expected",
        "trace_repeated_info_requests",
        "trace_guidance_precision",
    ]:
        if col in df.columns:
            label = col.replace("trace_", "").replace("_", " ").title()
            val = df[col].mean()
            if val != 0.0 or domain == "telecom" or "guidance" not in col:
                print(f"    {label:30s} {val:>8.2f} (avg)")

    print("\n" + "=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tau2-trace",
        description="tau2-TRACE: Trajectory-Aware Comprehensive Evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- analyze subcommand ----
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Post-hoc analysis of existing simulation results",
    )
    analyze_parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="Path to a tau2-bench Results JSON file",
    )
    analyze_parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/experiments/tau2_trace/results/tau2_trace_augmented.csv"),
        help="Output path for augmented CSV",
    )
    analyze_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Override domain (auto-detected from results if not specified)",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_results(
            results_path=args.results_file,
            output_path=args.output,
            domain_override=args.domain,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
