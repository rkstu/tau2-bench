# tau2-TRACE: Trajectory-Aware Comprehensive Evaluation

> Deterministic, zero-cost process-level metrics for tau2-bench — answering not just *"did the agent succeed?"* but *"how did it get there?"*

## Problem and Motivation

The current tau2-bench evaluation framework relies primarily on the binary `pass^k` metric. While highly effective for measuring final outcomes, this approach masks critical operational realities for enterprise-grade deployments:

| Question | tau2-bench Today | tau2-TRACE Adds |
|---|---|---|
| Did the agent succeed? | Final DB state match (`pass^k`) | Preserved — we don't replace this |
| Were tools called correctly? | Presence check (`ActionEvaluator`) | **Ordering** check against workflow DAGs + **read-after-write** verification |
| Was communication clear? | Substring match (`CommunicateEvaluator`) | **Guidance precision** (domain-specific terminology) + **token-to-action** ratio |
| Was the agent efficient? | Not measured | **Turns vs. expected**, action density, loop detection, redundancy detection |
| Why did the agent fail? | `reward=0` — no detail | Process diagnosis via error bursts, recovery pairs, and signature error classification |
| Is the agent robust? | Not measured | Opt-in **LLM semantic judge** for repeated-info and interaction fidelity detection |

## Proposed Solution

tau2-TRACE implements a **Layered Observability Architecture** that analyses existing `SimulationRun.messages` across three tiers:

```
                    SimulationRun.messages
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
  ┌──────────────────┐ ┌───────────┐ ┌──────────────────┐
  │ Layer 1:         │ │ Layer 2:  │ │ Layer 3:         │
  │ Efficiency       │ │ Adherence │ │ Interaction      │
  │ (Deterministic)  │ │ (Determ.) │ │ Quality          │
  │                  │ │           │ │                  │
  │ Redundancy       │ │ DAG order │ │ Action density   │
  │ Loops            │ │ Read-     │ │ Token ratio      │
  │ Error bursts     │ │ after-    │ │ Guidance prec.   │
  │ Recovery pairs   │ │ write     │ │ Repeated info    │
  │ Orphan tracking  │ │           │ │ ────────────     │
  │                  │ │           │ │ Opt-in LLM judge │
  └────────┬─────────┘ └─────┬─────┘ └────────┬─────────┘
           └─────────────────┼─────────────────┘
                             ▼
                   CompositeScorecard
                 (merged into DataFrame)
```

**Design constraints:**
- **Zero LLM calls by default** — all Layer 1–2 metrics are deterministic, O(n)
- **Opt-in semantic evaluation** — Layer 3 offers an LLM judge (`--llm-judge`) that hooks into `tau2.utils.llm_utils.generate()` for catching "semantic friction" that math cannot
- **Zero core changes** — lives entirely in `src/experiments/tau2_trace/`
- **Zero dependency additions** — uses only packages already in `pyproject.toml`
- **Domain-aware** — telecom gets DAG workflow checks; retail/airline get read-after-write verification

## Impact and Key Findings

Validated against real `gpt-4.1-mini` simulations, tau2-TRACE surfaces immediate, actionable intelligence:

- **Layer 1 — Cost vs. Reward Disconnect (Deterministic):** Identified passing agents (1.0 reward) that incurred 3x the API cost of more efficient counterparts due to severe turn overhead (14.0x expected actions) and loop entrapment. This highlights the hidden margin erosion masked by binary outcomes-based scoring.
- **Layer 2 — "Process-Perfect" Failures (Deterministic):** Successfully isolated failing agents (0.0 reward) that executed flawless DAG logic and zero redundant calls, but failed on a single final parameter. By grouping *Error Bursts* and tracking recovery pairs, TRACE narrows the debug scope instantly from "reasoning collapse" to "minor parameter fix."
- **Layer 3 — Semantic Friction Detection (Opt-In LLM Judge):** By toggling `--llm-judge`, tau2-TRACE extends evaluation beyond rigid math to catch interaction fidelity issues — agents repeatedly requesting known information or providing vague guidance instead of specific, actionable steps.

**What this looks like in practice** — same simulation, before and after:

```
# tau2-bench alone:
task_id=telecom_042  reward=1.0  agent_cost=$0.018

# tau2-bench + tau2-TRACE:
task_id=telecom_042  reward=1.0  agent_cost=$0.018
  trace_turns_vs_expected=14.0x  trace_redundant_tool_calls=1
  trace_matched_workflow=path2_mobile_data  trace_guidance_precision=0.875
  trace_error_recovery_rate=1.0  trace_error_burst_count=0
  trace_orphan_tool_messages=0  trace_read_after_write_score=1.0
```

See [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md) for detailed case studies, test results, and developer takeaways.

## Quick Start

```bash
# Analyse an existing simulation (deterministic, zero-cost, no API key)
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json

# With configurable error recovery window
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json \
    --recovery-window 5

# With opt-in LLM semantic judge (requires API key)
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json \
    --llm-judge --llm-judge-model gpt-4.1
```

Output: CSV with all standard tau2-bench columns plus `trace_*` columns, ready for pandas analysis.

## Technical Implementation

- **Pydantic models** throughout — consistent with `tau2.data_model.*` conventions
- **Error recovery redesign** — consecutive failures on the same tool are grouped into *Error Bursts* rather than inflating error counts; each recovery returns a `RecoveryPair(failed, recovered)` for diff inspection; signature errors (wrong function name) allow cross-name recovery
- **Dict comparison via `get_dict_hash`** from `tau2.utils.utils` — handles key ordering, nested structures, and non-standard types reliably
- **Orphan tool message tracking** — unmatched ToolMessages are logged via `loguru` and surfaced as `trace_orphan_tool_messages` rather than silently absorbed
- **Token-to-action ratio** — prefers actual `AssistantMessage.usage["completion_tokens"]` when available, falls back to character count as proxy
- **LLM judge integration** — lazy-imports `tau2.utils.llm_utils.generate()`, using the same API routing, rate limiting, and caching as the core benchmark; gracefully falls back to deterministic evaluation on any failure
- **Adversarial testing** — `AdversarialSimulatorWrapper` injects seeded interruptions and self-corrections via the Proxy Pattern — zero core code touched

## Verification

- **68/68 tests passing** across 6 test files — unit tests, integration tests with file I/O, end-to-end DataFrame merge validation, and mock-verified LLM judge tests
- **`ruff check`** — all lint checks passed
- **`ruff format`** — all formatting verified
- **Zero files modified** outside `src/experiments/tau2_trace/`
- Pre-computed result CSVs from real-world runs included in `results/`

## File Structure

```
src/experiments/tau2_trace/
├── models.py                  # Pydantic: ToolCallRecord, RecoveryPair, ErrorBurst, CompositeScorecard
├── trajectory_analyzer.py     # Parse messages → efficiency metrics (redundancy, loops, burst recovery)
├── tool_order_evaluator.py    # DAG workflow ordering (telecom) + read-after-write (all domains)
├── interaction_quality.py     # Action density, token ratio, guidance precision, LLM judge option
├── domain_router.py           # Domain-aware dispatch → CompositeScorecard
├── adversarial_wrapper.py     # UserSimulator proxy (interruptions, self-corrections)
├── run_experiment.py          # CLI: analyse existing Results JSON → augmented CSV
├── results/                   # Pre-computed CSVs from real-world validation
├── tests/                     # 68 tests (6 files)
├── README.md
└── EXAMPLE_OUTPUT.md          # Case studies, test results, reproduction steps
```

---

<details>
<summary><strong>Metrics Reference</strong> (click to expand)</summary>

### Trajectory Efficiency

| Metric | Definition |
|---|---|
| `trace_redundant_tool_calls` | Consecutive calls with identical name + arguments (via `get_dict_hash`) |
| `trace_loop_count` | Repeating tool call patterns (window=3) |
| `trace_error_count` | Raw count of tool calls that returned errors |
| `trace_error_recovery_rate` | Errors followed by successful retry within configurable window |
| `trace_error_burst_count` | Consecutive same-tool failures grouped into logical error events |
| `trace_error_bursts_recovered` | How many bursts ended with a successful recovery |
| `trace_orphan_tool_messages` | ToolMessages that could not be matched to a pending call (indicates run-level error) |

### Policy Adherence

| Metric | Definition |
|---|---|
| `trace_policy_adherence` | Telecom: tool transitions following DOT workflow DAG. Retail/Airline: read-after-write score |
| `trace_matched_workflow` | Best-matching telecom path (no_service / mobile_data / mms) |
| `trace_read_after_write_score` | Write operations verified by subsequent read |

### Interaction Quality

| Metric | Definition |
|---|---|
| `trace_action_density` | `tool_calls / turns` — higher = more efficient |
| `trace_token_to_action_ratio` | Agent tokens per tool call (prefers actual usage data, falls back to char count) |
| `trace_turns_vs_expected` | `actual_turns / expected_actions` — closer to 1.0 = better |
| `trace_guidance_precision` | (Telecom) Fraction of messages with specific device/network terms |
| `trace_repeated_info_requests` | Agent asks for data already returned by prior tool calls |

### Domain Routing

| Domain | DAG | Read-After-Write | Guidance | LLM Judge |
|---|---|---|---|---|
| Telecom | 3 DOT files | Yes | Yes | Opt-in |
| Retail | — | Yes | — | Opt-in |
| Airline | — | Yes | — | Opt-in |

</details>

<details>
<summary><strong>Limitations & Future Work</strong> (click to expand)</summary>

- **Telecom-only DAG**: Retail/airline use simpler read-after-write checks. Authoring DAGs for those policies would extend coverage.
- **LLM judge is opt-in**: The semantic layer adds fidelity but incurs API cost. Recommended for nightly or release-candidate evaluations, not every CI run.
- **Adversarial wrapper is illustrative**: Production deployment would need calibration against real user distributions.
- **Signature error patterns are conservative**: The classifier targets known error message formats; novel LLM error surfaces may require pattern additions.

</details>

<details>
<summary><strong>Related Work</strong> (click to expand)</summary>

Draws on: TRAJECT-Bench (tool usage diagnostics), TRACE (evidence banks), Agent GPA (temporal phase evaluation), SWE-eval (info-gain metrics). Adapted for tau2-bench's Dec-POMDP dual-control environment with a deterministic-first philosophy.

</details>
