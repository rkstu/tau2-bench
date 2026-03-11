# tau2-TRACE: Case Studies, Test Verification, and Impact

> From actual tau2-bench simulations (`gpt-4.1-mini`, Sierra orchestrator) and a comprehensive 68-test suite covering deterministic analysis, error recovery intelligence, and opt-in LLM semantic evaluation.

---

## Case Study 1: Telecom — "The Expensive Pass"

**Task**: Mobile data issue (airplane mode on + roaming disabled)
**tau2-bench**: reward=1.0, all assertions pass. **A perfect score.**

tau2-TRACE shows the cost behind that score:

| Finding | Value | Why It Matters |
|---|---|---|
| **28 turns** for a 2-action task | `turns_vs_expected = 14.0x` | At scale, this agent erodes margins under outcomes-based pricing |
| **13 total tool calls** (7 agent, 6 user) | vs 2 expected actions | Heavy API consumption for a simple fix |
| **1 redundant call** detected | `check_status_bar` called twice with same args (verified via `get_dict_hash`) | Wasted compute — deterministically flagged |
| **0 error bursts** | No consecutive failures | Robust execution, just inefficient |
| **0 orphan tool messages** | Every ToolMessage matched a pending call | Clean run — no simulation-level anomalies |
| **Workflow: `path2_mobile_data`** matched | From existing DOT files | Agent chose the correct troubleshooting path |
| **87.5% guidance precision** | 7 of 8 messages used specific terms | "Toggle airplane mode", "check roaming" — actionable instructions |
| **$0.018 cost** | vs $0.006 for the failing retail agent | 3x more expensive despite both having zero process errors |

**Developer action**: Use `trace_turns_vs_expected` as a regression metric. This agent works but needs efficiency optimization before production deployment.

---

## Case Study 2: Retail — "The Process-Perfect Failure"

**Task**: Order item exchange
**tau2-bench**: reward=0.0, exchange action failed. **A total failure.**

tau2-TRACE shows the process was actually sound:

| Finding | Value | Why It Matters |
|---|---|---|
| **0 redundant calls** | Clean tool discipline | No wasted API calls |
| **0 loops, 0 errors, 0 error bursts** | Flawless execution path | The agent's reasoning was correct throughout |
| **Correct tool sequence** | find_user → get_order → get_product ×2 | Policy adherence = 1.0 |
| **Read-after-write verified** | cancel → get_order_details | Mutation was properly confirmed |
| **2.6x turn overhead** | 13 turns vs 5 expected actions | Within acceptable range for enterprise deployment |
| **$0.006 cost** | 3x cheaper than the passing telecom agent | Superior unit economics |
| **Failure isolated** to `exchange_delivered_order_items` | Everything before it passed | A parameter issue, not a reasoning breakdown |

**Developer action**: Fix the exchange tool parameters. Don't redesign the agent's planning — the process is already production-quality. This diagnosis narrows debugging scope from "why did the agent fail?" to "why did this one tool call fail?"

---

## The Diagnostic Gap

| | Telecom (PASS) | Retail (FAIL) |
|---|---|---|
| **tau2-bench** | 1.0 | 0.0 |
| **Turn overhead** | 14.0x | 2.6x |
| **Cost** | $0.018 | $0.006 |
| **Process errors** | 0 | 0 |
| **Error bursts** | 0 | 0 |
| **Redundancy** | 1 | 0 |

`pass^k` says one agent is perfect and the other is worthless. tau2-TRACE says the "failing" agent has better process quality, costs less, and is one parameter fix away from production. **That's the diagnostic gap this experiment fills.**

---

## Test Suite: Verification and Results

### Overview

The tau2-TRACE test suite covers every layer of the architecture — from atomic metric computations through end-to-end file I/O with real `Results` objects. Tests run without API keys (LLM judge tests use `unittest.mock`).

```
68 passed, 0 failed, 1.46s
```

### Test Breakdown by File

| File | Tests | What It Validates |
|---|---|---|
| `test_trajectory_analyzer.py` | 21 | Message parsing, redundancy (with `get_dict_hash`), loop detection, error burst grouping, recovery pairs, signature error classification, orphan counting |
| `test_interaction_quality.py` | 18 | Action density, token-to-action ratio (with `msg.usage` preference), repeated info detection, guidance precision, LLM judge mock integration |
| `test_tool_order_evaluator.py` | 13 | DAG phase ordering (all 3 telecom paths), read-after-write verification, domain-specific routing |
| `test_domain_router.py` | 6 | End-to-end scorecard generation for telecom/retail, parameter threading, batch evaluation |
| `test_integration.py` | 5 | Full pipeline with realistic multi-turn simulations, Results JSON save/load, CSV export, DataFrame column verification |
| `test_adversarial_wrapper.py` | 5 | Passthrough, perturbation injection, seeded reproducibility, self-correction delivery, state reset |

### Key Test Scenarios and What They Prove

**Error Burst Intelligence** (`test_error_burst_grouping`):
Constructs a trajectory where the same tool fails 3 consecutive times then succeeds. Verifies that tau2-TRACE groups these into 1 logical burst (not 3 independent errors), marks the burst as recovered, and returns the `RecoveryPair(failed, recovered)` for diff inspection.

**Signature Error Recovery** (`test_signature_error_recovers_from_different_name`):
Constructs a trajectory where the agent calls `get_usr` (non-existent tool, error message: "unknown tool 'get_usr'"), then corrects to `get_user` (success). Verifies that tau2-TRACE classifies this as a signature error and allows cross-name recovery — something the original same-name-only check would miss.

**Dict Hash Reliability** (`test_nested_dict_args_hashed_correctly`):
Constructs two ToolCallRecords with arguments `{"a": 1, "b": {"c": 2}}` and `{"b": {"c": 2}, "a": 1}`. Verifies that `get_dict_hash` correctly identifies these as identical regardless of key ordering, preventing false negatives in redundancy detection.

**Token Usage Preference** (`test_prefers_actual_token_usage`):
Constructs AssistantMessages with both `content` (100/200 chars) and `usage` (50/80 completion_tokens). Verifies the ratio uses actual tokens (65.0) rather than character count (150.0), ensuring accurate measurement when LiteLLM provides usage data.

**Orphan Tracking** (`test_orphan_tool_message_counted`):
Sends a ToolMessage with an ID that matches no pending call. Verifies it is counted as an orphan (not silently swallowed) — flagging simulation-level anomalies that indicate run errors.

**LLM Judge Integration** (`test_repeated_info_llm_calls_generate`):
Mocks `tau2.utils.llm_utils.generate()` to return `{"count": 1}`. Verifies that the LLM judge code path correctly invokes the project's native generation API and parses the structured JSON response. Confirms the opt-in semantic layer hooks into the exact same API routing, rate limiting, and caching as the core benchmark.

**LLM Judge Graceful Degradation** (`test_llm_judge_fallback_on_failure`):
Mocks `generate()` to raise `Exception("API error")`. Verifies that the LLM judge falls back to deterministic evaluation rather than crashing — ensuring the `--llm-judge` flag is safe for CI/CD without risking pipeline failures.

**End-to-End File I/O** (`test_full_analyze_results_cli`):
Builds a full `Results` object with 3 telecom simulations, saves to JSON, runs the CLI `analyze_results` function, writes augmented CSV, reads it back, and verifies all `trace_*` columns are present and populated.

### Test Output (Abridged)

```
test_adversarial_wrapper.py    5/5  PASSED
test_domain_router.py          6/6  PASSED
test_integration.py            5/5  PASSED
test_interaction_quality.py   18/18 PASSED
test_tool_order_evaluator.py  13/13 PASSED
test_trajectory_analyzer.py   21/21 PASSED
─────────────────────────────────────
                              68/68 PASSED  (1.46s)
```

---

## Impact Summary

### For Enterprise Deployment (Sierra's Vision)

| Failure Mode | What Detects It | Layer |
|---|---|---|
| Agent passes but destroys margins | `trace_turns_vs_expected`, `trace_agent_cost` | Deterministic |
| Agent breaks troubleshooting workflow | `trace_policy_adherence`, `trace_matched_workflow` | Deterministic |
| Agent retries blindly after errors | `trace_error_burst_count`, `trace_error_bursts_recovered` | Deterministic |
| Agent uses non-existent tool names | Signature error classification + cross-name `RecoveryPair` | Deterministic |
| Agent re-asks for known information | `trace_repeated_info_requests` | Deterministic or LLM |
| Agent gives vague instead of specific guidance | `trace_guidance_precision` | Deterministic or LLM |
| Simulation has unmatched tool messages | `trace_orphan_tool_messages` | Deterministic |

### For Developers (CI/CD Integration)

- **Fast, free checks on every commit**: Layers 1–2 run in O(n) with zero API calls
- **Deep semantic checks on release candidates**: Layer 3 LLM judge runs on-demand via `--llm-judge`
- **Regression detection**: `trace_turns_vs_expected` and `trace_error_burst_count` as alerting thresholds
- **Debugging acceleration**: `RecoveryPair` diffs and error burst grouping narrow failure scope instantly

---

## Reproduction

```bash
cd tau2-bench
uv venv .venv --python 3.12 && uv pip install -e "."

# Full test suite (no API key required, ~1.5 seconds)
.venv/bin/python -m pytest src/experiments/tau2_trace/tests/ -v

# Real simulation (requires OPENAI_API_KEY)
.venv/bin/tau2 run --domain telecom --num-tasks 1 --num-trials 1 \
    --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini --save-to my_run

# Deterministic analysis
.venv/bin/python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json

# With LLM semantic judge (requires OPENAI_API_KEY)
.venv/bin/python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json \
    --llm-judge --llm-judge-model gpt-4.1
```

Pre-computed results from real-world runs are in `results/`.
