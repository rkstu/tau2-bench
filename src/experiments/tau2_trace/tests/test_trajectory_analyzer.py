"""Tests for the trajectory analyzer."""

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason

from experiments.tau2_trace.trajectory_analyzer import (
    _count_redundant_calls,
    _detect_loops,
    _compute_error_recovery,
    analyze_trajectory,
    extract_tool_calls,
)
from experiments.tau2_trace.models import ToolCallRecord


def _make_sim(messages, task_id="test_1", trial=0, agent_cost=0.5) -> SimulationRun:
    return SimulationRun(
        id="sim-1",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=messages,
        trial=trial,
        agent_cost=agent_cost,
    )


class TestExtractToolCalls:
    def test_extracts_agent_tool_calls(self):
        messages = [
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="tc1", name="get_user", arguments={"id": "123"})
                ],
            ),
            ToolMessage(id="tc1", role="tool", content='{"name": "Alice"}'),
        ]
        records = extract_tool_calls(messages)
        assert len(records) == 1
        assert records[0].name == "get_user"
        assert records[0].requestor == "assistant"
        assert records[0].result_content == '{"name": "Alice"}'
        assert records[0].result_error is False

    def test_extracts_user_tool_calls(self):
        messages = [
            UserMessage(
                role="user",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="check_status_bar",
                        arguments={},
                        requestor="user",
                    )
                ],
            ),
            ToolMessage(id="tc2", role="tool", content="No Service"),
        ]
        records = extract_tool_calls(messages)
        assert len(records) == 1
        assert records[0].requestor == "user"

    def test_marks_errors(self):
        messages = [
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="tc3", name="bad_call", arguments={"x": 1})],
            ),
            ToolMessage(id="tc3", role="tool", content="Error", error=True),
        ]
        records = extract_tool_calls(messages)
        assert records[0].result_error is True


class TestRedundantCalls:
    def test_no_redundancy(self):
        records = [
            ToolCallRecord(0, "tool_a", {"x": 1}, "assistant"),
            ToolCallRecord(1, "tool_b", {"y": 2}, "assistant"),
        ]
        assert _count_redundant_calls(records) == 0

    def test_detects_redundancy(self):
        records = [
            ToolCallRecord(0, "tool_a", {"x": 1}, "assistant"),
            ToolCallRecord(1, "tool_a", {"x": 1}, "assistant"),
            ToolCallRecord(2, "tool_a", {"x": 1}, "assistant"),
        ]
        assert _count_redundant_calls(records) == 2

    def test_different_args_not_redundant(self):
        records = [
            ToolCallRecord(0, "tool_a", {"x": 1}, "assistant"),
            ToolCallRecord(1, "tool_a", {"x": 2}, "assistant"),
        ]
        assert _count_redundant_calls(records) == 0


class TestLoopDetection:
    def test_no_loops(self):
        records = [ToolCallRecord(i, f"tool_{i}", {}, "assistant") for i in range(6)]
        assert _detect_loops(records) == 0

    def test_detects_loop(self):
        names = ["a", "b", "c", "a", "b", "c"]
        records = [ToolCallRecord(i, n, {}, "assistant") for i, n in enumerate(names)]
        assert _detect_loops(records, window=3) == 1


class TestErrorRecovery:
    def test_full_recovery(self):
        records = [
            ToolCallRecord(0, "tool_a", {}, "assistant", result_error=True),
            ToolCallRecord(1, "tool_a", {}, "assistant", result_error=False),
        ]
        errors, recovered, rate = _compute_error_recovery(records)
        assert errors == 1
        assert recovered == 1
        assert rate == 1.0

    def test_no_recovery(self):
        records = [
            ToolCallRecord(0, "tool_a", {}, "assistant", result_error=True),
            ToolCallRecord(1, "tool_b", {}, "assistant", result_error=False),
        ]
        errors, recovered, rate = _compute_error_recovery(records)
        assert errors == 1
        assert recovered == 0
        assert rate == 0.0

    def test_no_errors(self):
        records = [
            ToolCallRecord(0, "tool_a", {}, "assistant", result_error=False),
        ]
        errors, recovered, rate = _compute_error_recovery(records)
        assert errors == 0
        assert rate == 1.0


class TestAnalyzeTrajectory:
    def test_basic_analysis(self):
        messages = [
            AssistantMessage(role="assistant", content="How can I help?"),
            UserMessage(role="user", content="Fix my phone"),
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="get_customer_by_phone",
                        arguments={"phone": "555"},
                    )
                ],
            ),
            ToolMessage(id="t1", role="tool", content='{"id": "C1"}'),
            AssistantMessage(role="assistant", content="I found your account."),
        ]
        sim = _make_sim(messages)
        metrics, records = analyze_trajectory(sim, "telecom")

        assert metrics.task_id == "test_1"
        assert metrics.domain == "telecom"
        assert metrics.agent_message_count == 3
        assert metrics.user_message_count == 1
        assert metrics.agent_tool_call_count == 1
        assert metrics.total_tool_call_count == 1
        assert metrics.redundant_tool_calls == 0
        assert metrics.agent_cost == 0.5
