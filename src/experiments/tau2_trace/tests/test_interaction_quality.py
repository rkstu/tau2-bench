"""Tests for the interaction quality evaluator."""

import pytest

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

from experiments.tau2_trace.interaction_quality import (
    _compute_action_density,
    _compute_guidance_precision,
    _compute_token_to_action_ratio,
    _count_repeated_info_requests,
    evaluate_interaction_quality,
    TELECOM_GUIDANCE_TERMS,
)
from experiments.tau2_trace.models import ToolCallRecord


class TestActionDensity:
    def test_zero_turns(self):
        assert _compute_action_density(5, 0) == 0.0

    def test_normal(self):
        assert _compute_action_density(4, 8) == 0.5

    def test_high_density(self):
        assert _compute_action_density(10, 5) == 2.0


class TestTokenToActionRatio:
    def test_no_tool_calls(self):
        messages = [AssistantMessage(role="assistant", content="Hello there")]
        assert _compute_token_to_action_ratio(messages, 0) == 0.0

    def test_normal_ratio(self):
        messages = [
            AssistantMessage(role="assistant", content="A" * 100),
            AssistantMessage(role="assistant", content="B" * 200),
        ]
        ratio = _compute_token_to_action_ratio(messages, 2)
        assert ratio == 150.0  # 300 chars / 2 calls


class TestRepeatedInfoRequests:
    def test_no_repeats(self):
        messages = [
            ToolMessage(id="t1", role="tool", content='{"customer_id": "C12345"}'),
            AssistantMessage(role="assistant", content="I will help you with your order."),
        ]
        count = _count_repeated_info_requests(messages, [])
        assert count == 0

    def test_detects_repeat(self):
        messages = [
            ToolMessage(id="t1", role="tool", content='{"customer_id": "C12345"}'),
            AssistantMessage(
                role="assistant",
                content="What is your customer ID? Can you provide C12345 again?",
            ),
        ]
        count = _count_repeated_info_requests(messages, [])
        assert count >= 1


class TestGuidancePrecision:
    def test_precise_guidance(self):
        messages = [
            AssistantMessage(role="assistant", content="Please toggle your airplane mode off."),
            AssistantMessage(role="assistant", content="Now check your APN settings."),
        ]
        precision = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)
        assert precision == 1.0

    def test_vague_guidance(self):
        messages = [
            AssistantMessage(role="assistant", content="Please try the thing I mentioned."),
            AssistantMessage(role="assistant", content="Can you check that setting?"),
        ]
        precision = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)
        assert precision == 0.0

    def test_mixed_guidance(self):
        messages = [
            AssistantMessage(role="assistant", content="Please restart your device."),
            AssistantMessage(role="assistant", content="Just try again."),
        ]
        precision = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)
        assert precision == 0.5


class TestEvaluateInteractionQuality:
    def test_full_evaluation(self):
        messages = [
            AssistantMessage(role="assistant", content="How can I help?"),
            UserMessage(role="user", content="My phone has no service"),
            AssistantMessage(role="assistant", content="Please check your SIM card."),
            UserMessage(role="user", content="OK done"),
        ]
        records = [ToolCallRecord(1, "check_sim_status", {}, "user")]

        metrics = evaluate_interaction_quality(
            messages=messages,
            records=records,
            domain="telecom",
            task_id="test_1",
            trial=0,
            total_turns=4,
            agent_tool_call_count=0,
            expected_actions=2,
        )

        assert metrics.action_density == 0.25  # 1 tool / 4 turns
        assert metrics.turns_to_resolution == 4
        assert metrics.turns_vs_expected_ratio == 2.0  # 4 turns / 2 expected
        assert metrics.guidance_precision > 0.0  # "SIM card" matches
