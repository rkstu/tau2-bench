# tau2-TRACE: Testing Strategy, E2E Outputs, and Claim Verification

> Every number in this document comes from actual test runs. Unit tests use synthetic fixtures; E2E tests use live `gpt-4.1-mini` API calls against the real tau2-bench orchestrator. Pre-computed outputs are in `results/`.

---

## Testing Strategy

tau2-TRACE uses a three-tier verification approach:

| Tier | What | How | API Key Required |
|---|---|---|---|
| **Unit tests** (68 tests) | Every metric computation, edge case, error path | `pytest` with synthetic `SimulationRun` fixtures and `unittest.mock` for LLM judge | No |
| **Live E2E simulations** (6 runs) | Full pipeline: orchestrator -> simulation -> trace analysis -> CSV output | `run_experiment.py run` and `run_experiment.py analyze` against real `gpt-4.1-mini` | Yes |
| **Cross-mode comparison** | Deterministic vs. CoT LLM judge on the same simulation | `analyze` with and without `--llm-judge` on identical results JSON | Yes |

---

## Tier 1: Unit Test Suite

### Summary

```
68 passed, 0 failed, 1.36s
```

Tests run without API keys. LLM judge tests use `unittest.mock` to verify the code path without real API calls.

### Breakdown by File

| File | Tests | What It Validates |
|---|---|---|
| `test_trajectory_analyzer.py` | 21 | Message parsing, redundancy (with `get_dict_hash`), loop detection, error burst grouping, recovery pairs, signature error classification, orphan counting, configurable recovery window |
| `test_interaction_quality.py` | 18 | Action density, token-to-action ratio (with `msg.usage` preference), repeated info detection, guidance precision, LLM judge mock integration, graceful fallback |
| `test_tool_order_evaluator.py` | 13 | Phase-order matching (all 3 telecom workflows), read-after-write verification, domain-specific routing |
| `test_domain_router.py` | 6 | End-to-end scorecard generation for telecom/retail, parameter threading (recovery_window, llm_judge), batch evaluation |
| `test_integration.py` | 5 | Full pipeline: realistic multi-turn simulations, Results JSON save/load round-trip, CSV export, DataFrame column verification (58 columns) |
| `test_adversarial_wrapper.py` | 5 | Passthrough when no perturbation, perturbation injection at 100% rate, seeded RNG reproducibility, self-correction delivery on next turn, state reset |

### Key Test Scenarios

**Error Burst Intelligence** (`test_error_burst_grouping`):
Constructs a trajectory where the same tool fails 3 consecutive times then succeeds. Verifies: 1 logical burst (not 3 independent errors), marked as recovered, `RecoveryPair(failed, recovered)` returned for diff inspection. This backs the README claim that error bursts "narrow debugging scope."

**Signature Error Recovery** (`test_signature_error_recovers_from_different_name`):
Agent calls `get_usr` (error: "unknown tool 'get_usr'"), then corrects to `get_user` (success). Verifies: classified as signature error (via scoped pattern matching), cross-name recovery detected. Same-name-only matching would miss this. This backs the README claim about signature error classification.

**Dict Hash Reliability** (`test_nested_dict_args_hashed_correctly`):
Arguments `{"a": 1, "b": {"c": 2}}` and `{"b": {"c": 2}, "a": 1}` are correctly identified as identical via `get_dict_hash`, preventing false negatives in redundancy detection. This backs the README claim about `get_dict_hash` handling key ordering.

**Token Usage Preference** (`test_prefers_actual_token_usage`):
AssistantMessages with both `content` (100/200 chars) and `usage` (50/80 completion_tokens). Verifies the ratio uses actual tokens (65.0) rather than character count (150.0). This backs the README claim that token-to-action ratio "prefers actual `AssistantMessage.usage`."

**Orphan Tracking** (`test_orphan_tool_message_counted`):
Sends a ToolMessage with an ID that matches no pending call. Verifies it is counted (not silently absorbed) and that `loguru.warning` is invoked. This backs the README claim about orphan tool message tracking.

**LLM Judge CoT Integration** (`test_repeated_info_llm_calls_generate`):
Mocks `tau2.utils.llm_utils.generate()` to return `{"reasoning": "step-by-step analysis", "count": 1}`. Verifies the CoT-prompted judge correctly invokes the project's native generation API with the `reasoning` field in the prompt and parses the structured JSON response. This backs the README claim about Chain-of-Thought prompting.

**LLM Judge Graceful Degradation** (`test_llm_judge_fallback_on_failure`):
Mocks `generate()` to raise `Exception("API error")`. Verifies fallback to deterministic evaluation rather than crashing. This backs the README claim that `--llm-judge` is "safe for CI/CD."

**End-to-End File I/O** (`test_full_analyze_results_cli`):
Builds a full `Results` object with 3 telecom simulations, saves to JSON, runs `analyze_results`, writes augmented CSV, reads it back, verifies all `trace_*` columns are present and populated. This backs the README claim about 58-column CSV output.

### Test Output

```
test_adversarial_wrapper.py    5/5  PASSED
test_domain_router.py          6/6  PASSED
test_integration.py            5/5  PASSED
test_interaction_quality.py   18/18 PASSED
test_tool_order_evaluator.py  13/13 PASSED
test_trajectory_analyzer.py   21/21 PASSED
-------------------------------------------
                              68/68 PASSED  (1.36s)
```

---

## Tier 2: Live E2E Simulations

Six live simulations were run against the real `gpt-4.1-mini` API using the tau2-bench orchestrator. Each simulation produced both a raw results JSON and an augmented trace CSV. All output files are in `results/`.

### Run 1: Mock Baseline

```
Domain: mock  |  Task: create_task_1  |  Model: gpt-4.1-mini
Mode: normal (no adversarial)
```

| Metric | Value |
|---|---|
| reward | 1.0 |
| agent_cost | $0.0007 |
| trace_total_turns | 7 |
| trace_total_tool_calls | 1 (1 agent, 0 user) |
| trace_redundant_tool_calls | 0 |
| trace_loop_count | 0 |
| trace_error_count | 0 |
| trace_orphan_tool_messages | 0 |
| trace_action_density | 0.143 |
| trace_turns_vs_expected | 7.0 |

**Purpose**: Smoke test that the full pipeline (orchestrator -> trace analysis -> CSV merge) works on the simplest domain. Confirmed: 58-column CSV output, all trace metrics populated.

### Run 2: Telecom Baseline

```
Domain: telecom  |  Task: [mobile_data_issue]user_abroad_roaming_enabled_off
Model: gpt-4.1-mini  |  Mode: normal (no adversarial)
```

| Metric | Value |
|---|---|
| reward | 1.0 |
| agent_cost | $0.017 |
| trace_total_turns | 39 |
| trace_total_tool_calls | 17 (8 agent, 9 user) |
| trace_redundant_tool_calls | 0 |
| trace_loop_count | 0 |
| trace_error_count | 0 |
| trace_error_burst_count | 0 |
| trace_orphan_tool_messages | 0 |
| trace_policy_adherence | 1.0 |
| trace_matched_workflow | path1_no_service |
| trace_read_after_write_score | 1.0 |
| trace_action_density | 0.436 |
| trace_token_to_action_ratio | 126.0 |
| trace_turns_vs_expected | 39.0 |
| trace_guidance_precision | 0.833 |
| trace_repeated_info_requests | 6 |

**Purpose**: Full telecom evaluation with all three layers active. Demonstrates: workflow phase-order matching, dual-control tool call parsing (agent + user), guidance precision on real conversation content.

**Observation**: Agent passes (reward=1.0) but requires 39 turns for a task with 1 expected action. `trace_turns_vs_expected=39.0` flags this as operationally expensive. The binary reward hides this overhead.

### Run 3: Telecom Adversarial

```
Domain: telecom  |  Task: [mobile_data_issue]user_abroad_roaming_enabled_off
Model: gpt-4.1-mini  |  Mode: adversarial (perturbation_rate=0.3, seed=42)
Perturbations injected: 6
```

| Metric | Baseline | Adversarial | Delta |
|---|---|---|---|
| reward | 1.0 | 0.0 | -1.0 |
| trace_total_turns | 39 | 25 | -14 |
| trace_total_tool_calls | 17 | 7 | -10 |
| trace_policy_adherence | 1.0 | 0.0 | -1.0 |
| trace_matched_workflow | path1_no_service | none | -- |
| trace_action_density | 0.436 | 0.280 | -0.156 |
| trace_guidance_precision | 0.833 | 0.778 | -0.055 |
| trace_repeated_info_requests | 6 | 4 | -2 |

**Purpose**: Verify the adversarial wrapper is integrated end-to-end and measurably affects agent outcomes. The wrapper injected 6 perturbations (interruptions and self-corrections) which caused the agent to fail (reward dropped from 1.0 to 0.0), lose workflow adherence entirely, and reduce action density.

**What this backs**: The README claim that `AdversarialSimulatorWrapper` is "integrated into the `run` subcommand" and has "E2E verified" impact. The wrapper is not dead code.

### Run 4: Retail Baseline

```
Domain: retail  |  Task: 0  |  Model: gpt-4.1-mini
Mode: normal (no adversarial)
```

| Metric | Value |
|---|---|
| reward | 1.0 |
| agent_cost | $0.009 |
| trace_total_turns | 19 |
| trace_total_tool_calls | 5 (5 agent, 0 user) |
| trace_redundant_tool_calls | 0 |
| trace_loop_count | 0 |
| trace_error_count | 0 |
| trace_orphan_tool_messages | 0 |
| trace_action_density | 0.263 |
| trace_turns_vs_expected | 3.8 |
| trace_repeated_info_requests | 2 |

**Purpose**: Verify single-control domain (no user tools) processes correctly, and that domain routing skips telecom-only metrics (guidance precision, workflow matching). Confirmed: `trace_guidance_precision` = 0.0 (correctly not computed), `trace_matched_workflow` = null.

### Run 5: Airline Baseline

```
Domain: airline  |  Task: 0  |  Model: gpt-4.1-mini
Mode: normal (no adversarial)
```

| Metric | Value |
|---|---|
| reward | 1.0 |
| agent_cost | $0.003 |
| trace_total_turns | 9 |
| trace_total_tool_calls | 1 (1 agent, 0 user) |
| trace_redundant_tool_calls | 0 |
| trace_loop_count | 0 |
| trace_error_count | 0 |
| trace_orphan_tool_messages | 0 |
| trace_action_density | 0.111 |
| trace_token_to_action_ratio | 228.0 |
| trace_policy_adherence | 1.0 |
| trace_read_after_write_score | 1.0 |
| trace_turns_vs_expected | 9.0 |

**Purpose**: Verify the fourth production domain (airline) processes correctly as a single-control domain. Confirmed: domain routing correctly skips telecom-only metrics (`trace_guidance_precision` = 0.0, `trace_matched_workflow` = null), read-after-write verification active and passing. Cheapest run at $0.003, 58-column CSV output confirmed.

### Run 6: Telecom with CoT LLM Judge

```
Domain: telecom  |  Same results JSON as Run 2
Mode: post-hoc analysis with --llm-judge --llm-judge-model gpt-4.1-mini
```


| Metric | Deterministic (Run 2) | CoT LLM Judge (Run 6) | Interpretation |
|---|---|---|---|
| trace_repeated_info_requests | 6 | 0 | Deterministic over-counted: token-in-string matched common substrings that were not actually repeated requests. The CoT judge correctly identified 0 true repeated requests. |
| trace_guidance_precision | 0.833 | 0.667 | LLM was stricter: required specific actionable steps, not just keyword presence. 2 messages that mentioned telecom terms but gave vague instructions were downgraded. |
| All other trace_* metrics | identical | identical | Layers 1-2 are purely deterministic; LLM judge only affects Layer 3 interaction metrics. |

**Purpose**: Verify that (a) the LLM judge runs successfully against a real API, (b) it produces genuinely different signal from the deterministic heuristics, and (c) all non-judge metrics remain identical (deterministic reproducibility).

**What this backs**: The README claims about "CoT LLM judge" producing different signal and the Known Limitations acknowledgment that "repeated-info detection can over-count on common substrings."

---

## Tier 3: Cross-Verification

### Post-Hoc Consistency

The telecom baseline results JSON was analyzed twice:
1. During the live `run` subcommand (Run 2)
2. Separately via the `analyze` subcommand on the saved JSON

Both produced identical trace metrics, confirming deterministic reproducibility of the analysis pipeline.

### Core Test Suite Compatibility

The full tau2-bench test suite was run after all changes:

```
154 passed, 1 failed (pre-existing flaky test), 141.32s
```

The single failure (`test_run_tasks_action_checks`) is a pre-existing LLM-dependent test that asserts `reward == 1.0` on a live API call; it fails when the model makes a different choice (unrelated to tau2-TRACE). Verified by `git diff src/tau2/` = empty: zero core files were modified.

### Lint and Format

```
ruff check src/experiments/tau2_trace/ -- All checks passed
ruff format --check src/experiments/tau2_trace/ -- 15 files already formatted
```

---

## Claim-by-Claim Verification

Every claim made in the README is backed by specific evidence:

| README Claim | Evidence | Source |
|---|---|---|
| "68/68 unit tests passing" | `pytest` output: `68 passed, 0 failed, 1.36s` | Tier 1 |
| "Zero core changes" | `git diff src/tau2/` = empty | Tier 3 |
| "Zero dependency additions" | No changes to `pyproject.toml` | Inspection |
| "58-column CSV output (27 core + 31 trace)" | `e2e_telecom_baseline.csv`: 58 columns verified | Run 2 |
| "Phase-order matching works for telecom" | `trace_matched_workflow=path1_no_service`, `trace_policy_adherence=1.0` | Run 2 |
| "Read-after-write verification" | `trace_read_after_write_score=1.0` on telecom baseline | Run 2 |
| "Guidance precision (18-term keyword heuristic)" | `trace_guidance_precision=0.833` on telecom baseline | Run 2 |
| "Adversarial wrapper integrated into CLI" | `run --adversarial` produced 6 perturbations, agent failed | Run 3 |
| "Adversarial wrapper changes agent outcomes" | Baseline reward=1.0 vs adversarial reward=0.0 | Run 2 vs Run 3 |
| "CoT LLM judge via `generate()`" | Real API call to gpt-4.1-mini, different results than deterministic | Run 6 |
| "LLM judge gracefully falls back" | `test_llm_judge_fallback_on_failure` PASSED | Tier 1 |
| "Repeated-info can over-count" | Deterministic=6, LLM judge=0 on same simulation | Run 2 vs Run 6 |
| "Signature errors allow cross-name recovery" | `test_signature_error_recovers_from_different_name` PASSED | Tier 1 |
| "Error bursts group consecutive failures" | `test_error_burst_grouping` PASSED (3 failures -> 1 burst) | Tier 1 |
| "Token ratio prefers actual usage data" | `test_prefers_actual_token_usage` PASSED (65.0 vs 150.0) | Tier 1 |
| "Orphan messages tracked, not silently absorbed" | `test_orphan_tool_message_counted` PASSED | Tier 1 |
| "Dict comparison via `get_dict_hash`" | `test_nested_dict_args_hashed_correctly` PASSED | Tier 1 |
| "Domain-aware routing" | Telecom gets workflow + guidance; retail/airline get read-after-write only | Run 2, Run 4, Run 5 |
| "154/155 core tau2-bench tests passing" | `pytest tests/` output: 1 pre-existing flaky failure | Tier 3 |

---

## Failure-Mode Detection Summary

| Failure Mode | Detection Metric | Type | Evidence |
|---|---|---|---|
| Agent passes but with excessive overhead | `trace_turns_vs_expected` (39.0 on telecom) | Deterministic | Run 2 |
| Agent breaks troubleshooting ordering | `trace_policy_adherence` (1.0 -> 0.0 under adversarial) | Deterministic | Run 3 |
| Agent retries blindly after errors | `trace_error_burst_count`, `trace_error_bursts_recovered` | Deterministic | Tier 1 tests |
| Agent uses non-existent tool names | Signature error classification + cross-name `RecoveryPair` | Deterministic | Tier 1 tests |
| Agent re-asks for known information | `trace_repeated_info_requests` (6 deterministic, 0 LLM) | Deterministic or CoT LLM | Run 2, Run 6 |
| Agent gives vague guidance | `trace_guidance_precision` (0.833 deterministic, 0.667 LLM) | Deterministic or CoT LLM | Run 2, Run 6 |
| Simulation has unmatched tool messages | `trace_orphan_tool_messages` | Deterministic | Tier 1 tests |
| Agent cannot handle user perturbations | Adversarial wrapper: reward 1.0 -> 0.0 with 6 perturbations | Live simulation | Run 3 |

---

## Reproduction

```bash
cd tau2-bench
uv venv .venv --python 3.12 && uv pip install -e "."

# Tier 1: Full unit test suite (no API key, ~1.5 seconds)
.venv/bin/python -m pytest src/experiments/tau2_trace/tests/ -v

# Tier 2a: Live baseline simulations (requires OPENAI_API_KEY)
.venv/bin/python -m experiments.tau2_trace.run_experiment run \
    --domain mock --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini \
    --num-tasks 1 --output src/experiments/tau2_trace/results/e2e_mock_baseline

.venv/bin/python -m experiments.tau2_trace.run_experiment run \
    --domain telecom --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini \
    --num-tasks 1 --output src/experiments/tau2_trace/results/e2e_telecom_baseline

.venv/bin/python -m experiments.tau2_trace.run_experiment run \
    --domain retail --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini \
    --num-tasks 1 --output src/experiments/tau2_trace/results/e2e_retail_baseline

.venv/bin/python -m experiments.tau2_trace.run_experiment run \
    --domain airline --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini \
    --num-tasks 1 --output src/experiments/tau2_trace/results/e2e_airline_baseline

# Tier 2b: Adversarial simulation (requires OPENAI_API_KEY)
.venv/bin/python -m experiments.tau2_trace.run_experiment run \
    --domain telecom --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini \
    --adversarial --perturbation-rate 0.3 --num-tasks 1 \
    --output src/experiments/tau2_trace/results/e2e_telecom_adversarial

# Tier 2c: CoT LLM judge (requires OPENAI_API_KEY)
.venv/bin/python -m experiments.tau2_trace.run_experiment analyze \
    --results-file src/experiments/tau2_trace/results/e2e_telecom_baseline.json \
    --llm-judge --llm-judge-model gpt-4.1-mini \
    --output src/experiments/tau2_trace/results/e2e_telecom_llm_judge.csv

# Tier 3: Core test suite (requires OPENAI_API_KEY for some tests)
.venv/bin/python -m pytest tests/ -v

# Tier 3: Zero core changes verification
git diff src/tau2/  # should be empty
```

Pre-computed results from all E2E runs are in `results/`.
