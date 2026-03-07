"""
Phase 1: Core Trajectory Analyzer.

Parses SimulationRun.messages into structured ToolCallRecords and computes
deterministic efficiency metrics (redundancy, loops, error recovery).
"""

from __future__ import annotations

from typing import Optional

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun

from experiments.tau2_trace.models import ToolCallRecord, TrajectoryMetrics


def extract_tool_calls(messages: list[Message]) -> list[ToolCallRecord]:
    """
    Walk the message list and pair each ToolCall with its corresponding
    ToolMessage result. Returns an ordered list of ToolCallRecords.
    """
    records: list[ToolCallRecord] = []
    pending: dict[str, ToolCallRecord] = {}
    turn_index = 0

    for msg in messages:
        if isinstance(msg, (AssistantMessage, UserMessage)):
            if msg.role == "assistant":
                turn_index += 1
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    record = ToolCallRecord(
                        turn_index=turn_index,
                        name=tc.name,
                        arguments=dict(tc.arguments) if tc.arguments else {},
                        requestor=tc.requestor,
                    )
                    if tc.id:
                        pending[tc.id] = record
                    records.append(record)

        elif isinstance(msg, ToolMessage):
            if msg.id and msg.id in pending:
                pending[msg.id].result_content = msg.content
                pending[msg.id].result_error = bool(msg.error)
            else:
                # Orphan tool message -- still capture it on the last record
                if records:
                    records[-1].result_content = msg.content
                    records[-1].result_error = bool(msg.error)

        elif isinstance(msg, MultiToolMessage):
            for sub in msg.tool_messages:
                if sub.id and sub.id in pending:
                    pending[sub.id].result_content = sub.content
                    pending[sub.id].result_error = bool(sub.error)

    return records


def _count_redundant_calls(records: list[ToolCallRecord]) -> int:
    """
    Count consecutive tool calls with the same name and arguments
    where no state-changing event occurred in between.
    """
    if len(records) < 2:
        return 0

    redundant = 0
    for i in range(1, len(records)):
        prev, curr = records[i - 1], records[i]
        if (
            curr.name == prev.name
            and curr.arguments == prev.arguments
            and curr.requestor == prev.requestor
        ):
            redundant += 1
    return redundant


def _detect_loops(records: list[ToolCallRecord], window: int = 3) -> int:
    """
    Detect loops: a repeating sequence of tool names of length >= window.
    Returns the number of detected loop occurrences.
    """
    if len(records) < window * 2:
        return 0

    names = [r.name for r in records]
    loops = 0
    i = 0
    while i <= len(names) - window * 2:
        pattern = tuple(names[i : i + window])
        next_segment = tuple(names[i + window : i + window * 2])
        if pattern == next_segment:
            loops += 1
            i += window
        else:
            i += 1
    return loops


def _compute_error_recovery(records: list[ToolCallRecord]) -> tuple[int, int, float]:
    """
    For each tool call that returned an error, check if a subsequent call
    (within 3 steps) succeeded with the same or a related tool.
    Returns (error_count, recovered_count, recovery_rate).
    """
    errors = 0
    recovered = 0
    for i, rec in enumerate(records):
        if rec.result_error:
            errors += 1
            lookahead = records[i + 1 : i + 4]
            for future in lookahead:
                if not future.result_error and future.name == rec.name:
                    recovered += 1
                    break

    rate = recovered / errors if errors > 0 else 1.0
    return errors, recovered, rate


def analyze_trajectory(
    simulation: SimulationRun,
    domain: str,
) -> tuple[TrajectoryMetrics, list[ToolCallRecord]]:
    """
    Analyze a single SimulationRun and produce trajectory metrics.

    Returns:
        A tuple of (TrajectoryMetrics, list of ToolCallRecords) so downstream
        evaluators can reuse the parsed tool calls.
    """
    messages = simulation.messages
    records = extract_tool_calls(messages)

    agent_msgs = sum(1 for m in messages if isinstance(m, AssistantMessage))
    user_msgs = sum(1 for m in messages if isinstance(m, UserMessage))
    tool_msgs = sum(
        1
        for m in messages
        if isinstance(m, ToolMessage) or isinstance(m, MultiToolMessage)
    )

    agent_tc = [r for r in records if r.requestor == "assistant"]
    user_tc = [r for r in records if r.requestor == "user"]

    redundant = _count_redundant_calls(records)
    loops = _detect_loops(records)
    errors, recovered, recovery_rate = _compute_error_recovery(records)

    metrics = TrajectoryMetrics(
        task_id=simulation.task_id,
        trial=simulation.trial,
        domain=domain,
        total_turns=agent_msgs + user_msgs,
        agent_message_count=agent_msgs,
        user_message_count=user_msgs,
        tool_message_count=tool_msgs,
        agent_tool_call_count=len(agent_tc),
        user_tool_call_count=len(user_tc),
        total_tool_call_count=len(records),
        redundant_tool_calls=redundant,
        loop_count=loops,
        error_count=errors,
        errors_recovered=recovered,
        error_recovery_rate=recovery_rate,
        agent_cost=simulation.agent_cost,
    )

    return metrics, records
