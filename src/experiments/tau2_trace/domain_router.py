"""
Phase 4: Domain-Aware Metric Router.

Dispatches to the correct evaluators based on domain and aggregates
results into a CompositeScorecard.
"""

from __future__ import annotations

from typing import Optional

from tau2.data_model.simulation import SimulationRun

from experiments.tau2_trace.interaction_quality import evaluate_interaction_quality
from experiments.tau2_trace.models import CompositeScorecard
from experiments.tau2_trace.tool_order_evaluator import evaluate_tool_ordering
from experiments.tau2_trace.trajectory_analyzer import analyze_trajectory


def evaluate_simulation_trace(
    simulation: SimulationRun,
    domain: str,
    expected_actions: int = 0,
) -> CompositeScorecard:
    """
    Run the full tau2-TRACE analysis pipeline on a single SimulationRun.

    Args:
        simulation: A completed simulation run with full message trajectory.
        domain: The domain name (telecom, retail, airline).
        expected_actions: Expected number of actions from task metadata
            (task_num_actions from Results.to_df()). Used for turns_vs_expected.

    Returns:
        A CompositeScorecard containing all computed metrics.
    """
    # Phase 1: Core trajectory analysis
    traj_metrics, records = analyze_trajectory(simulation, domain)

    # Phase 2: Tool ordering evaluation (domain-aware)
    ordering_metrics = evaluate_tool_ordering(
        records=records,
        domain=domain,
        task_id=simulation.task_id,
        trial=simulation.trial,
    )

    # Phase 3: Interaction quality (domain-aware)
    interaction_metrics = evaluate_interaction_quality(
        messages=simulation.messages,
        records=records,
        domain=domain,
        task_id=simulation.task_id,
        trial=simulation.trial,
        total_turns=traj_metrics.total_turns,
        agent_tool_call_count=traj_metrics.agent_tool_call_count,
        expected_actions=expected_actions,
    )

    return CompositeScorecard(
        task_id=simulation.task_id,
        trial=simulation.trial,
        domain=domain,
        trajectory=traj_metrics,
        ordering=ordering_metrics,
        interaction=interaction_metrics,
    )


def evaluate_results_trace(
    simulations: list[SimulationRun],
    domain: str,
    task_expected_actions: Optional[dict[str, int]] = None,
) -> list[CompositeScorecard]:
    """
    Batch-evaluate a list of simulation runs.

    Args:
        simulations: List of SimulationRun objects.
        domain: Domain name.
        task_expected_actions: Optional mapping of task_id -> expected action count.

    Returns:
        List of CompositeScorecard, one per simulation.
    """
    if task_expected_actions is None:
        task_expected_actions = {}

    scorecards = []
    for sim in simulations:
        expected = task_expected_actions.get(sim.task_id, 0)
        scorecard = evaluate_simulation_trace(sim, domain, expected)
        scorecards.append(scorecard)

    return scorecards
