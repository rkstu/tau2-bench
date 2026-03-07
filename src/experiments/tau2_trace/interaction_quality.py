"""
Phase 3: Interaction Quality Evaluator.

Computes deterministic, mathematical metrics about agent-user interaction
quality. No NLP, no regex on free-form text, no LLM calls.
"""

from __future__ import annotations

from typing import Optional

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolMessage,
    UserMessage,
)

from experiments.tau2_trace.models import InteractionMetrics, ToolCallRecord

# Known telecom user tool parameter terms that indicate specific guidance.
# When an agent mentions these terms while instructing the user, it shows
# precise, actionable guidance rather than vague instructions.
TELECOM_GUIDANCE_TERMS = {
    "airplane mode",
    "airplane",
    "apn",
    "sim card",
    "sim",
    "mobile data",
    "data roaming",
    "roaming",
    "speed test",
    "data saver",
    "network mode",
    "vpn",
    "wi-fi calling",
    "wifi calling",
    "mms",
    "reboot",
    "restart",
    "status bar",
    "payment",
}


def _compute_action_density(
    total_tool_calls: int,
    total_turns: int,
) -> float:
    """Ratio of tool calls executed to conversational turns."""
    if total_turns == 0:
        return 0.0
    return total_tool_calls / total_turns


def _compute_token_to_action_ratio(
    messages: list[Message],
    agent_tool_call_count: int,
) -> float:
    """
    Estimate agent tokens per tool call.
    Uses character count as a proxy when token usage isn't available.
    """
    if agent_tool_call_count == 0:
        return 0.0

    total_chars = 0
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.content:
            total_chars += len(msg.content)

    return total_chars / agent_tool_call_count


def _count_repeated_info_requests(
    messages: list[Message],
    records: list[ToolCallRecord],
) -> int:
    """
    Count agent messages that ask for information already present in
    prior tool results. We check if any substantial substring (>= 6 chars)
    from a tool result appears in a later agent question-like message.
    """
    tool_result_tokens: set[str] = set()
    repeated = 0

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            # Extract identifiers and values from tool results
            for token in msg.content.split():
                cleaned = token.strip("\"',{}[]:()")
                if len(cleaned) >= 6:
                    tool_result_tokens.add(cleaned.lower())

        elif isinstance(msg, AssistantMessage) and msg.content:
            if not tool_result_tokens:
                continue
            # Check if agent is asking about something already known
            content_lower = msg.content.lower()
            has_question = "?" in msg.content or any(
                w in content_lower
                for w in ("what is", "can you", "could you", "please provide")
            )
            if has_question:
                for token in tool_result_tokens:
                    if token in content_lower:
                        repeated += 1
                        break

    return repeated


def _compute_guidance_precision(
    messages: list[Message],
    guidance_terms: set[str],
) -> float:
    """
    For telecom: fraction of agent messages that contain specific
    tool-related parameter terms when instructing the user.
    Higher = agent gives more specific, actionable instructions.
    """
    agent_msgs = [
        msg
        for msg in messages
        if isinstance(msg, AssistantMessage) and msg.content and not msg.is_tool_call()
    ]

    if not agent_msgs:
        return 0.0

    precise_count = 0
    for msg in agent_msgs:
        content_lower = msg.content.lower()
        if any(term in content_lower for term in guidance_terms):
            precise_count += 1

    return precise_count / len(agent_msgs)


def evaluate_interaction_quality(
    messages: list[Message],
    records: list[ToolCallRecord],
    domain: str,
    task_id: str,
    trial: Optional[int] = None,
    total_turns: int = 0,
    agent_tool_call_count: int = 0,
    expected_actions: int = 0,
) -> InteractionMetrics:
    """
    Compute interaction quality metrics for a single simulation.

    Args:
        messages: Full message trajectory from SimulationRun.
        records: Parsed ToolCallRecords from trajectory_analyzer.
        domain: Domain name (telecom/retail/airline).
        task_id: Task identifier.
        trial: Trial number.
        total_turns: Pre-computed turn count from TrajectoryMetrics.
        agent_tool_call_count: Pre-computed agent tool call count.
        expected_actions: Expected number of actions from task metadata.
    """
    total_tool_calls = len(records)

    action_density = _compute_action_density(total_tool_calls, total_turns)
    token_to_action = _compute_token_to_action_ratio(messages, agent_tool_call_count)
    repeated = _count_repeated_info_requests(messages, records)

    turns_vs_expected = 0.0
    if expected_actions > 0:
        turns_vs_expected = total_turns / expected_actions

    guidance = 0.0
    if domain == "telecom":
        guidance = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)

    return InteractionMetrics(
        task_id=task_id,
        trial=trial,
        action_density=round(action_density, 4),
        token_to_action_ratio=round(token_to_action, 2),
        turns_to_resolution=total_turns,
        turns_vs_expected_ratio=round(turns_vs_expected, 4),
        repeated_info_requests=repeated,
        guidance_precision=round(guidance, 4),
    )
