"""Data models for tau2-TRACE metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCallRecord:
    """A single tool call extracted from the trajectory."""

    turn_index: int
    name: str
    arguments: dict
    requestor: str  # "assistant" or "user"
    result_content: Optional[str] = None
    result_error: bool = False


@dataclass
class TrajectoryMetrics:
    """Deterministic efficiency metrics computed from a simulation trajectory."""

    task_id: str
    trial: Optional[int]
    domain: str

    total_turns: int = 0
    agent_message_count: int = 0
    user_message_count: int = 0
    tool_message_count: int = 0

    agent_tool_call_count: int = 0
    user_tool_call_count: int = 0
    total_tool_call_count: int = 0

    redundant_tool_calls: int = 0
    loop_count: int = 0
    error_count: int = 0
    errors_recovered: int = 0
    error_recovery_rate: float = 0.0

    agent_cost: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "trace_domain": self.domain,
            "trace_total_turns": self.total_turns,
            "trace_agent_messages": self.agent_message_count,
            "trace_user_messages": self.user_message_count,
            "trace_tool_messages": self.tool_message_count,
            "trace_agent_tool_calls": self.agent_tool_call_count,
            "trace_user_tool_calls": self.user_tool_call_count,
            "trace_total_tool_calls": self.total_tool_call_count,
            "trace_redundant_tool_calls": self.redundant_tool_calls,
            "trace_loop_count": self.loop_count,
            "trace_error_count": self.error_count,
            "trace_errors_recovered": self.errors_recovered,
            "trace_error_recovery_rate": self.error_recovery_rate,
            "trace_agent_cost": self.agent_cost,
        }


@dataclass
class OrderingMetrics:
    """Policy adherence metrics from DAG-based tool ordering evaluation."""

    task_id: str
    trial: Optional[int]

    policy_adherence_score: float = 0.0
    total_transitions: int = 0
    valid_transitions: int = 0
    policy_breaches: list[tuple[str, str]] = field(default_factory=list)
    matched_workflow: Optional[str] = None
    read_after_write_score: float = 0.0
    write_calls_total: int = 0
    write_calls_verified: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "trace_policy_adherence": self.policy_adherence_score,
            "trace_total_transitions": self.total_transitions,
            "trace_valid_transitions": self.valid_transitions,
            "trace_policy_breach_count": len(self.policy_breaches),
            "trace_matched_workflow": self.matched_workflow,
            "trace_read_after_write_score": self.read_after_write_score,
            "trace_write_calls_total": self.write_calls_total,
            "trace_write_calls_verified": self.write_calls_verified,
        }


@dataclass
class InteractionMetrics:
    """Deterministic interaction quality metrics."""

    task_id: str
    trial: Optional[int]

    action_density: float = 0.0
    token_to_action_ratio: float = 0.0
    turns_to_resolution: int = 0
    turns_vs_expected_ratio: float = 0.0
    repeated_info_requests: int = 0
    guidance_precision: float = 0.0  # telecom only

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "trace_action_density": self.action_density,
            "trace_token_to_action_ratio": self.token_to_action_ratio,
            "trace_turns_to_resolution": self.turns_to_resolution,
            "trace_turns_vs_expected": self.turns_vs_expected_ratio,
            "trace_repeated_info_requests": self.repeated_info_requests,
            "trace_guidance_precision": self.guidance_precision,
        }


@dataclass
class CompositeScorecard:
    """Aggregated scorecard for a single simulation run."""

    task_id: str
    trial: Optional[int]
    domain: str

    trajectory: Optional[TrajectoryMetrics] = None
    ordering: Optional[OrderingMetrics] = None
    interaction: Optional[InteractionMetrics] = None

    def to_dict(self) -> dict:
        result: dict = {
            "task_id": self.task_id,
            "trial": self.trial,
            "trace_domain": self.domain,
        }
        if self.trajectory:
            result.update(
                {k: v for k, v in self.trajectory.to_dict().items() if k not in result}
            )
        if self.ordering:
            result.update(
                {k: v for k, v in self.ordering.to_dict().items() if k not in result}
            )
        if self.interaction:
            result.update(
                {k: v for k, v in self.interaction.to_dict().items() if k not in result}
            )
        return result
