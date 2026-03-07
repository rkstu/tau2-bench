# tau2-TRACE: Trajectory-Aware Comprehensive Evaluation

> **Deterministic, zero-cost trajectory observability for tau2-bench.**

## Motivation

tau2-bench evaluates conversational AI agents using a binary outcome: does the final database state match the ground truth? While mathematically rigorous via `pass^k`, this approach reveals nothing about **how** the agent solved (or failed) the task:

- An agent that looped through redundant API calls but stumbled into the correct state scores 100%.
- Two agents with identical `pass^k` scores cannot be differentiated by efficiency or safety.
- The `ActionEvaluator` checks tool call **presence** but not **ordering** — a policy violation that happens to produce the right result goes undetected.
- The `CommunicateEvaluator` uses substring matching (with a `TODO: This could be improved!` in the source).

tau2-TRACE adds **process-level observability** by analyzing the full interaction trajectory (`SimulationRun.messages`). All metrics are deterministic and rule-based — no LLM calls, no API costs, O(n) computation.

## Architecture

```
┌───────────────────────────────────────────────────┐
│              Existing tau2-bench pipeline           │
│  run_task() → Orchestrator → evaluate_simulation() │
│  Output: SimulationRun with pass^k reward           │
└───────────────────────┬───────────────────────────┘
                        │ SimulationRun.messages
                        ▼
┌───────────────────────────────────────────────────┐
│              tau2-TRACE (this experiment)           │
│                                                     │
│  trajectory_analyzer.py  → TrajectoryMetrics        │
│  tool_order_evaluator.py → OrderingMetrics          │
│  interaction_quality.py  → InteractionMetrics       │
│  domain_router.py        → CompositeScorecard       │
│                                                     │
│  Output: Augmented DataFrame (core + trace metrics) │
└───────────────────────────────────────────────────┘
```

**Zero core changes.** All code lives in `src/experiments/tau2_trace/`. The existing evaluation pipeline is untouched.

## Metrics

### Trajectory Efficiency

| Metric | Description |
|---|---|
| `trace_redundant_tool_calls` | Consecutive tool calls with identical name + arguments |
| `trace_loop_count` | Repeating tool call patterns (window=3) |
| `trace_error_count` | Tool calls that returned errors |
| `trace_error_recovery_rate` | Fraction of errors followed by successful retry |
| `trace_agent_tool_calls` | Total agent-initiated tool calls |
| `trace_user_tool_calls` | Total user-initiated tool calls (telecom) |

### Policy Adherence

| Metric | Description |
|---|---|
| `trace_policy_adherence` | Fraction of tool transitions following valid workflow order (telecom: uses DOT DAG files) |
| `trace_matched_workflow` | Best-matching telecom workflow path (path1/path2/path3) |
| `trace_read_after_write_score` | Fraction of write operations verified by a subsequent read |
| `trace_policy_breach_count` | Number of out-of-order tool transitions |

### Interaction Quality

| Metric | Description |
|---|---|
| `trace_action_density` | `total_tool_calls / total_turns` — higher means more efficient |
| `trace_token_to_action_ratio` | Agent characters generated per tool call — lower means less verbose |
| `trace_turns_vs_expected` | `actual_turns / expected_actions` — closer to 1.0 is better |
| `trace_repeated_info_requests` | Agent asks for information already in prior tool results |
| `trace_guidance_precision` | (Telecom only) Fraction of agent messages containing specific tool parameter terms |

### Domain-Aware Routing

| Domain | DAG Ordering | Read-After-Write | Guidance Precision |
|---|---|---|---|
| Telecom | Full workflow DAG (3 DOT files) | Yes | Yes |
| Retail | N/A | Yes (exchange, cancel, return) | No |
| Airline | N/A | Yes (book, cancel, update) | No |

## Quick Start

### Post-hoc Analysis (Recommended)

Analyze an existing simulation results file:

```bash
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/tau2/simulations/my_run.json \
    --output src/experiments/tau2_trace/results/tau2_trace_augmented.csv
```

This produces a CSV with all core tau2-bench columns plus `trace_*` columns.

### Adversarial User Simulator (Optional)

For stress-testing agent robustness with chaotic user behavior:

```python
from tau2.user.user_simulator import UserSimulator
from experiments.tau2_trace.adversarial_wrapper import AdversarialSimulatorWrapper

base_sim = UserSimulator(tools=tools, instructions=instructions, llm="gpt-4.1")
chaos_sim = AdversarialSimulatorWrapper(base_sim, perturbation_rate=0.2, seed=42)
# Pass chaos_sim to the Orchestrator in place of base_sim
```

Perturbation types:
- **Interruption** (configurable %): Replaces user response with an off-topic question
- **Self-correction** (configurable %): Lets the response through, then sends a correction next turn

All perturbations are seeded for reproducibility.

## Validated Against Real Sierra Data

tau2-TRACE has been tested end-to-end against **real tau2-bench simulations** (not just synthetic data):

| Domain | Task | Model | Reward | tau2-TRACE Key Finding |
|---|---|---|---|---|
| **Telecom** | Mobile data (airplane+roaming) | gpt-4.1-mini | 1.0 (PASS) | 28 turns / 13 tool calls for a 2-action task. `turns_vs_expected=14.0x`. Guidance precision: 87.5% |
| **Retail** | Order item exchange | gpt-4.1-mini | 0.0 (FAIL) | Process was correct (0 redundancy, 0 loops), failure was at final action. More efficient than the passing telecom agent |
| **Mock** | Task creation | gpt-4.1-mini | 1.0 (PASS) | Clean minimal execution. Action density: 0.20 |

**50 unit + integration tests**, all passing:

```
======================== 50 passed, 1 warning in 1.32s =========================
```

See [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md) for full metric breakdowns and cross-domain analysis.

## Running Tests

```bash
# Unit tests (no API key needed)
pytest src/experiments/tau2_trace/tests/ -v
```

## File Structure

```
src/experiments/tau2_trace/
├── README.md                  # This file
├── __init__.py
├── models.py                  # Dataclasses: TrajectoryMetrics, OrderingMetrics, etc.
├── trajectory_analyzer.py     # Parse messages → tool call records → efficiency metrics
├── tool_order_evaluator.py    # DAG-based ordering (telecom) + read-after-write (all)
├── interaction_quality.py     # Action density, token ratio, guidance precision
├── domain_router.py           # Domain-aware dispatch + CompositeScorecard
├── adversarial_wrapper.py     # UserSimulator proxy for chaos testing
├── run_experiment.py          # CLI entry point
└── tests/
    ├── test_trajectory_analyzer.py
    ├── test_tool_order_evaluator.py
    ├── test_interaction_quality.py
    ├── test_domain_router.py
    └── test_adversarial_wrapper.py
```

## Example: Real-World Evidence of the "High-Score Illusion"

From actual `gpt-4.1-mini` simulations run against the Sierra tau2-bench orchestrator:

### A passing telecom agent (reward=1.0) with hidden inefficiency

| Metric | Value | What tau2-bench Sees | What tau2-TRACE Reveals |
|---|---|---|---|
| `reward` | 1.0 | PASS | — |
| `trace_turns_vs_expected` | **14.0x** | — | 28 turns for a 2-action task — massive overhead |
| `trace_total_tool_calls` | **13** | — | 13 API calls when 2 were needed |
| `trace_guidance_precision` | **0.875** | — | 87.5% of instructions used specific telecom terms |
| `trace_matched_workflow` | `path2_mobile_data` | — | Correctly followed the mobile data troubleshooting path |
| `trace_agent_cost` | **$0.018** | $0.018 | 3x the cost of the retail agent |

### A failing retail agent (reward=0.0) that was actually well-behaved

| Metric | Value | What tau2-bench Sees | What tau2-TRACE Reveals |
|---|---|---|---|
| `reward` | 0.0 | FAIL | — |
| `trace_redundant_tool_calls` | **0** | — | Zero wasted calls — process was sound |
| `trace_turns_vs_expected` | **2.6x** | — | Reasonable overhead for the task |
| `trace_error_count` | **0** | — | No tool errors — failure was at the final action, not the process |
| `trace_agent_cost` | **$0.006** | $0.006 | 3x cheaper than the passing agent |

**In an enterprise deployment with outcomes-based pricing (Sierra's model), the "passing" telecom agent costs 3x more per resolution. tau2-TRACE makes this visible. `pass^k` alone cannot.**

See [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md) for the complete cross-domain analysis.

## Limitations & Future Work

- **Telecom-only DAG evaluation**: Retail and airline use simpler read-after-write checks since no DOT workflow files exist for those domains. Manually authoring DAGs for retail/airline policy would extend coverage.
- **No LLM-based semantic checks**: By design, all metrics are deterministic. The existing `NLAssertionsEvaluator` pattern could be used to add semantic checks (e.g., hallucination detection) as a future Layer 2.
- **Adversarial wrapper is experimental**: The perturbation types are illustrative. Production deployment would need calibration against real user behavior distributions.

## Related Work

This experiment draws on concepts from:
- **TRAJECT-Bench**: Fine-grained tool usage diagnostics via trajectory alignment
- **TRACE**: Dynamic evidence banks for multi-dimensional process evaluation
- **Agent GPA**: Temporal workflow phase evaluation (Goal, Plan, Action)
- **SWE-eval**: Info-gain metrics penalizing redundant tool executions

Adapted to tau2-bench's Dec-POMDP dual-control environment with a strict deterministic-first philosophy.
