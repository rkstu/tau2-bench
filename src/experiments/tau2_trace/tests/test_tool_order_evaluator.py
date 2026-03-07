"""Tests for the tool ordering evaluator."""

import pytest

from experiments.tau2_trace.models import ToolCallRecord
from experiments.tau2_trace.tool_order_evaluator import (
    _best_workflow_match,
    _check_read_after_write,
    _match_workflow_phase_order,
    evaluate_tool_ordering,
    RETAIL_WRITE_TO_READ,
    TELECOM_PATH1_PHASE_ORDER,
)


class TestPhaseOrderMatching:
    def test_perfect_order(self):
        tools = ["check_status_bar", "toggle_airplane_mode", "check_sim_status", "reset_apn_settings", "get_line_details"]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert valid == total
        assert total == 4  # 5 phases matched = 4 transitions

    def test_reversed_order(self):
        tools = ["get_line_details", "reset_apn_settings", "check_sim_status", "toggle_airplane_mode", "check_status_bar"]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert valid == 0

    def test_partial_match(self):
        tools = ["check_status_bar", "reset_apn_settings"]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert total == 1
        assert valid == 1

    def test_no_matching_tools(self):
        tools = ["unrelated_tool_a", "unrelated_tool_b"]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert total == 0


class TestBestWorkflowMatch:
    def test_matches_path1(self):
        tools = ["check_status_bar", "toggle_airplane_mode", "check_sim_status"]
        name, score, valid, total = _best_workflow_match(tools)
        assert name == "path1_no_service"
        assert score > 0.0

    def test_matches_path2(self):
        tools = ["run_speed_test", "toggle_roaming", "toggle_data", "toggle_data_saver_mode"]
        name, score, valid, total = _best_workflow_match(tools)
        assert name == "path2_mobile_data"

    def test_matches_path3(self):
        tools = ["can_send_mms", "check_status_bar", "toggle_wifi_calling", "reset_apn_settings"]
        name, score, valid, total = _best_workflow_match(tools)
        assert name == "path3_mms"


class TestReadAfterWrite:
    def test_verified_write(self):
        records = [
            ToolCallRecord(0, "cancel_pending_order", {"id": "O1"}, "assistant"),
            ToolCallRecord(1, "get_order_details", {"id": "O1"}, "assistant"),
        ]
        total, verified, score = _check_read_after_write(records, RETAIL_WRITE_TO_READ)
        assert total == 1
        assert verified == 1
        assert score == 1.0

    def test_unverified_write(self):
        records = [
            ToolCallRecord(0, "cancel_pending_order", {"id": "O1"}, "assistant"),
            ToolCallRecord(1, "find_user_id_by_email", {"email": "x"}, "assistant"),
        ]
        total, verified, score = _check_read_after_write(records, RETAIL_WRITE_TO_READ)
        assert total == 1
        assert verified == 0
        assert score == 0.0

    def test_no_writes(self):
        records = [
            ToolCallRecord(0, "get_order_details", {"id": "O1"}, "assistant"),
        ]
        total, verified, score = _check_read_after_write(records, RETAIL_WRITE_TO_READ)
        assert total == 0
        assert score == 1.0


class TestEvaluateToolOrdering:
    def test_telecom_domain(self):
        records = [
            ToolCallRecord(0, "check_status_bar", {}, "user"),
            ToolCallRecord(1, "toggle_airplane_mode", {}, "user"),
            ToolCallRecord(2, "check_sim_status", {}, "user"),
        ]
        metrics = evaluate_tool_ordering(records, "telecom", "task_1", trial=0)
        assert metrics.matched_workflow == "path1_no_service"
        assert metrics.policy_adherence_score > 0.0

    def test_retail_domain(self):
        records = [
            ToolCallRecord(0, "cancel_pending_order", {"id": "O1"}, "assistant"),
            ToolCallRecord(1, "get_order_details", {"id": "O1"}, "assistant"),
        ]
        metrics = evaluate_tool_ordering(records, "retail", "task_1", trial=0)
        assert metrics.read_after_write_score == 1.0
        assert metrics.matched_workflow is None
