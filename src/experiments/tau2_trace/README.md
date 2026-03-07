# tau2-TRACE: Trajectory-Aware Comprehensive Evaluation

> **Deterministic, zero-cost trajectory observability for tau2-bench.**

## The Problem: The "High-Score Illusion"

tau2-bench evaluates agents with a binary question: *does the final database state match ground truth?*

This is mathematically rigorous (`pass^k`) but hides critical enterprise failures:

- **Inefficient Success**: An agent that loops 28 times to solve a 2-step task gets the same `reward=1.0` as an efficient agent.
- **Process Violations**: An agent that skips post-mutation verification but lands on the right state goes undetected.
- **Diagnostic Black Box**: When an agent fails, `pass^k` says "0" — it cannot explain *where* in the reasoning chain the break occurred.

### Real-World Proof

We ran tau2-TRACE against real `gpt-4.1-mini` simulations on the Sierra tau2-bench orchestrator. The results demonstrate that **binary rewards are deceptive**:

| Metric | Telecom Agent (**PASS**) | Retail Agent (**FAIL**) | What This Means |
|---|---|---|---|
| **tau2-bench reward** | 1.0 | 0.0 | *Standard result — no further insight* |
| **Turns vs. Expected** | **14.0x** | **2.6x** | The passing agent was wildly inefficient |
| **Agent Cost** | **$0.018** | **$0.006** | The passing agent cost 3x more |
| **Redundant Calls** | 1 | 0 | The failing agent had zero process errors |
| **Guidance Precision** | 87.5% | N/A | The passing agent at least guided well |

**The "passing" telecom agent is an enterprise liability. The "failing" retail agent had superior process quality.** In Sierra's outcomes-based pricing model, the passing agent costs 3x more per resolution. tau2-TRACE makes this visible. `pass^k` alone cannot.

---

## The Solution: Process-Level Observability

tau2-TRACE analyzes the full interaction trajectory (`SimulationRun.messages`) to answer not just *"did the agent succeed?"* but *"how did it get there?"*

### Design Principles

- **Zero LLM Judge Costs**: All metrics are deterministic and rule-based — O(n) computation.
- **Zero Core Changes**: Lives entirely in `src/experiments/tau2_trace/`. The existing evaluation pipeline is untouched.
- **Domain-Aware**: Uses the existing DOT workflow files for Telecom DAG adherence; Read-After-Write checks for Retail/Airline.
- **Orthogonal**: Runs *alongside* the standard `pass^k` pipeline, not inside it. No risk to existing leaderboards.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Existing tau2-bench pipeline                      │
│  tau2 run → Orchestrator → evaluate_simulation() → pass^k reward     │
│  Output: SimulationRun (messages, reward, cost)                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ SimulationRun.messages
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    tau2-TRACE (this experiment)                        │
│  Zero core changes. Zero LLM costs. Fully deterministic.              │
│                                                                        │
│  trajectory_analyzer.py  → Efficiency (redundancy, loops, recovery)   │
│  tool_order_evaluator.py → Adherence (DAG ordering, read-after-write) │
│  interaction_quality.py  → Quality (action density, guidance, cost)    │
│  domain_router.py        → Domain-aware dispatch + aggregation         │
│                                                                        │
│  Output: Augmented DataFrame (standard metrics + trace_* columns)     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What tau2-bench Evaluates Today vs. What tau2-TRACE Adds

| Dimension | tau2-bench (Before) | tau2-TRACE (After) |
|---|---|---|
| **Outcome** | Final DB state match (`pass^k`) | Preserved — tau2-TRACE does not replace this |
| **Tool Usage** | Were expected tools called? (presence only, `ActionEvaluator`) | Were they called in the **right order**? Were writes **verified**? Were calls **redundant**? |
| **Communication** | Did agent mention required info? (substring match, has `TODO: improve!`) | How **precise** was guidance? (telecom term usage). How **verbose** was the agent? (token-to-action ratio) |
| **Efficiency** | Not measured | Turns vs expected actions, action density, loop detection, error recovery rate |
| **Cost Insight** | `agent_cost` field exists but isn't evaluated | Cost contextualized against efficiency — same pass, different price |
| **Failure Diagnosis** | "reward=0" — no further detail | Was the process correct? Where did it break? Is the agent close to production-ready? |

---

## Metrics Reference

### Trajectory Efficiency

| Metric | Definition |
|---|---|
| `trace_redundant_tool_calls` | Consecutive calls with identical name + arguments (no state change between) |
| `trace_loop_count` | Repeating tool call patterns (sliding window, length 3) |
| `trace_error_count` | Tool calls that returned errors |
| `trace_error_recovery_rate` | Fraction of errors followed by successful retry within 3 steps |
| `trace_agent_tool_calls` | Total agent-initiated tool calls |
| `trace_user_tool_calls` | Total user-initiated tool calls (telecom dual-control) |

### Policy Adherence

| Metric | Definition |
|---|---|
| `trace_policy_adherence` | Telecom: fraction of tool transitions following the DOT workflow DAG. Retail/Airline: read-after-write score |
| `trace_matched_workflow` | Best-matching telecom workflow path (`path1_no_service`, `path2_mobile_data`, `path3_mms`) |
| `trace_read_after_write_score` | Fraction of write operations (cancel, exchange, etc.) verified by a subsequent read |
| `trace_policy_breach_count` | Number of out-of-order tool transitions against the DAG |

### Interaction Quality

| Metric | Definition |
|---|---|
| `trace_action_density` | `total_tool_calls / total_turns` — higher = more efficient |
| `trace_token_to_action_ratio` | Agent characters per tool call — lower = less verbose |
| `trace_turns_vs_expected` | `actual_turns / expected_actions` — closer to 1.0 = better |
| `trace_repeated_info_requests` | Agent asks for data already present in prior tool results |
| `trace_guidance_precision` | (Telecom only) Fraction of agent messages containing specific device/network terms |

### Domain-Aware Routing

| Domain | DAG Ordering | Read-After-Write | Guidance Precision |
|---|---|---|---|
| **Telecom** | 3 DOT workflow files (no_service, mobile_data, mms) | Yes | Yes (30 telecom-specific terms) |
| **Retail** | N/A | Yes (exchange, cancel, return, modify) | N/A |
| **Airline** | N/A | Yes (book, cancel, update, certificate) | N/A |

---

## Quick Start

### Post-hoc Analysis (Recommended — No Simulation Needed)

Analyze any existing tau2-bench results file:

```bash
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json \
    --output src/experiments/tau2_trace/results/tau2_trace_augmented.csv
```

Produces a CSV with all standard tau2-bench columns *plus* `trace_*` columns — ready for pandas/Jupyter analysis.

### Generate + Analyze (Requires API Key)

```bash
# Run a simulation
tau2 run --domain telecom --num-tasks 1 --num-trials 1 \
    --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini \
    --save-to my_telecom_run

# Analyze with tau2-TRACE
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_telecom_run.json
```

### Adversarial User Simulator (Optional Stress-Test)

```python
from tau2.user.user_simulator import UserSimulator
from experiments.tau2_trace.adversarial_wrapper import AdversarialSimulatorWrapper

base_sim = UserSimulator(tools=tools, instructions=instructions, llm="gpt-4.1")
chaos_sim = AdversarialSimulatorWrapper(base_sim, perturbation_rate=0.2, seed=42)
# Pass chaos_sim to the Orchestrator — same interface, chaotic behavior
```

Perturbation types (seeded for reproducibility):
- **Interruption**: Off-topic user question mid-troubleshooting
- **Self-correction**: User gives wrong info, then corrects next turn

---

## Validation

### Real-World Data (3 domains, real LLM trajectories)

| Domain | Task | Model | Reward | tau2-TRACE Finding |
|---|---|---|---|---|
| **Telecom** | Mobile data (airplane+roaming) | gpt-4.1-mini | 1.0 (PASS) | 14.0x turn overhead, $0.018 cost, 87.5% guidance precision |
| **Retail** | Order item exchange | gpt-4.1-mini | 0.0 (FAIL) | Zero process errors — failure at final tool call only |
| **Mock** | Task creation | gpt-4.1-mini | 1.0 (PASS) | Clean baseline — 5 turns, 1 tool call |

### Test Suite: 50/50 Passed

```
======================== 50 passed, 1 warning in 1.32s =========================
```

| Test File | Tests | Coverage |
|---|---|---|
| `test_trajectory_analyzer.py` | 11 | Parsing, redundancy, loops, error recovery |
| `test_tool_order_evaluator.py` | 13 | DAG ordering, workflow matching, read-after-write |
| `test_interaction_quality.py` | 11 | Action density, token ratio, guidance precision |
| `test_domain_router.py` | 5 | Cross-domain routing, scorecard serialization |
| `test_adversarial_wrapper.py` | 5 | Proxy pattern, seeded reproducibility |
| `test_integration.py` | 5 | End-to-end with file I/O, DataFrame merge |

See [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md) for the full case studies with detailed metric breakdowns.

---

## File Structure

```
src/experiments/tau2_trace/
├── README.md                        # This file
├── EXAMPLE_OUTPUT.md                # Real-world case studies and validation
├── __init__.py
├── models.py                        # Dataclasses: TrajectoryMetrics, OrderingMetrics, etc.
├── trajectory_analyzer.py           # Parse messages → tool call records → efficiency metrics
├── tool_order_evaluator.py          # DAG-based ordering (telecom) + read-after-write (all)
├── interaction_quality.py           # Action density, token ratio, guidance precision
├── domain_router.py                 # Domain-aware dispatch + CompositeScorecard
├── adversarial_wrapper.py           # UserSimulator proxy for chaos testing
├── run_experiment.py                # CLI entry point
├── results/                         # Real-world analysis outputs (CSV)
│   ├── telecom_trace.csv
│   ├── retail_trace.csv
│   └── mock_trace.csv
└── tests/
    ├── test_trajectory_analyzer.py
    ├── test_tool_order_evaluator.py
    ├── test_interaction_quality.py
    ├── test_domain_router.py
    ├── test_adversarial_wrapper.py
    └── test_integration.py
```

---

## Limitations & Future Work

- **Telecom-only DAG evaluation**: Retail and airline use simpler read-after-write checks since no DOT workflow files exist for those domains. Authoring DAGs for retail/airline policies would extend full workflow coverage.
- **No LLM-based semantic checks**: By design, all metrics are deterministic. The existing `NLAssertionsEvaluator` pattern could be used to add semantic checks (e.g., hallucination detection) as a future Layer 2.
- **Adversarial wrapper is experimental**: The perturbation types are illustrative. Production deployment would need calibration against real user behavior distributions.

## Related Work

This experiment draws on concepts from:
- **TRAJECT-Bench**: Fine-grained tool usage diagnostics via trajectory alignment
- **TRACE**: Dynamic evidence banks for multi-dimensional process evaluation
- **Agent GPA**: Temporal workflow phase evaluation (Goal, Plan, Action)
- **SWE-eval**: Info-gain metrics penalizing redundant tool executions

Adapted to tau2-bench's Dec-POMDP dual-control environment with a strict deterministic-first philosophy.
