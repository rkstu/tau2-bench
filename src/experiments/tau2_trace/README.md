# tau2-TRACE: Trajectory-Aware Comprehensive Evaluation

> Deterministic, zero-cost process-level metrics for tau2-bench — answering not just *"did the agent succeed?"* but *"how did it get there?"*

## The Gap

tau2-bench's `pass^k` metric checks whether the final database state matches ground truth. This is rigorous but leaves critical questions unanswered:

| Question | tau2-bench Today | tau2-TRACE Adds |
|---|---|---|
| Did the agent succeed? | Final DB state match (`pass^k`) | Preserved — we don't replace this |
| Were tools called correctly? | Presence check (`ActionEvaluator`) | **Ordering** check against workflow DAGs + **read-after-write** verification |
| Was communication clear? | Substring match (`CommunicateEvaluator`) | **Guidance precision** (telecom-specific terminology) + **token-to-action** verbosity ratio |
| Was the agent efficient? | Not measured | **Turns vs. expected**, action density, loop detection, redundancy detection |
| Why did the agent fail? | `reward=0` — no detail | Process diagnosis: was the failure in reasoning, tool selection, or a single parameter? |

## Proof: Real Simulations, Counterintuitive Results

We ran tau2-TRACE against real `gpt-4.1-mini` trajectories from the Sierra orchestrator:

| | Telecom Agent | Retail Agent |
|---|---|---|
| **tau2-bench says** | **PASS** (reward=1.0) | **FAIL** (reward=0.0) |
| Turns for the task | 28 (expected: 2 actions) | 13 (expected: 5 actions) |
| Turn overhead | **14.0x** | **2.6x** |
| Cost | **$0.018** | **$0.006** |
| Redundant tool calls | 1 | 0 |
| Process errors | 0 | 0 |

The passing agent cost 3x more and was 5x less efficient. The failing agent had a flawless process — it broke at one tool call parameter. **`pass^k` treats these as 1.0 and 0.0. tau2-TRACE shows the full picture.**

See [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md) for the detailed case studies with developer-actionable takeaways.

## How It Works

tau2-TRACE reads `SimulationRun.messages` (the existing trajectory data) and computes metrics across three dimensions:

```
                    SimulationRun.messages
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌─────────────┐ ┌───────────┐ ┌────────────┐
      │ Efficiency   │ │ Adherence │ │ Quality    │
      │              │ │           │ │            │
      │ Redundancy   │ │ DAG order │ │ Action     │
      │ Loops        │ │ Read-     │ │ density    │
      │ Error        │ │ after-    │ │ Guidance   │
      │ recovery     │ │ write     │ │ precision  │
      └──────┬───────┘ └─────┬─────┘ └──────┬─────┘
             └───────────────┼───────────────┘
                             ▼
                   CompositeScorecard
                 (merged into DataFrame)
```

**What this looks like in practice** — same simulation, before and after:

```
# tau2-bench alone:
task_id=telecom_042  reward=1.0  agent_cost=$0.018

# tau2-bench + tau2-TRACE:
task_id=telecom_042  reward=1.0  agent_cost=$0.018
  trace_turns_vs_expected=14.0x  trace_redundant_tool_calls=1
  trace_matched_workflow=path2_mobile_data  trace_guidance_precision=0.875
  trace_error_recovery_rate=1.0  trace_read_after_write_score=1.0
```

Same data in, richer data out. Every `trace_*` column is computed deterministically from the existing `SimulationRun.messages` — no extra API calls, no changes to the simulation itself.

**Design constraints:**
- **Zero LLM calls** — all metrics are deterministic, O(n)
- **Zero core changes** — lives entirely in `src/experiments/tau2_trace/`
- **Domain-aware** — telecom gets DAG workflow checks (using the existing DOT files); retail/airline get read-after-write verification

Also includes an **adversarial user simulator wrapper** (`AdversarialSimulatorWrapper`) that injects seeded interruptions and self-corrections via the Proxy Pattern — no core code touched.

## Quick Start

```bash
# Analyze an existing simulation (no API key needed)
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json

# Or generate + analyze (requires OPENAI_API_KEY)
tau2 run --domain telecom --num-tasks 1 --num-trials 1 \
    --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini --save-to my_run
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json
```

Output: CSV with all standard tau2-bench columns plus `trace_*` columns, ready for pandas analysis.

## Validation

Tested against real Sierra orchestrator output across 3 domains (mock, telecom, retail). 50/50 tests passing — unit tests, integration tests with file I/O, and end-to-end DataFrame merge validation. Pre-computed result CSVs included in `results/`.

## File Structure

```
src/experiments/tau2_trace/
├── models.py                  # TrajectoryMetrics, OrderingMetrics, InteractionMetrics, CompositeScorecard
├── trajectory_analyzer.py     # Parse messages → efficiency metrics (redundancy, loops, recovery)
├── tool_order_evaluator.py    # DAG workflow ordering (telecom) + read-after-write (all domains)
├── interaction_quality.py     # Action density, token ratio, guidance precision
├── domain_router.py           # Domain-aware dispatch → CompositeScorecard
├── adversarial_wrapper.py     # UserSimulator proxy (interruptions, self-corrections)
├── run_experiment.py          # CLI: analyze existing Results JSON → augmented CSV
├── results/                   # Pre-computed CSVs from real-world validation
├── tests/                     # 50 tests (6 files)
├── README.md
└── EXAMPLE_OUTPUT.md          # Detailed case studies
```

---

<details>
<summary><strong>Metrics Reference</strong> (click to expand)</summary>

### Trajectory Efficiency

| Metric | Definition |
|---|---|
| `trace_redundant_tool_calls` | Consecutive calls with identical name + arguments |
| `trace_loop_count` | Repeating tool call patterns (window=3) |
| `trace_error_count` | Tool calls that returned errors |
| `trace_error_recovery_rate` | Errors followed by successful retry within 3 steps |

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
| `trace_token_to_action_ratio` | Agent characters per tool call — lower = less verbose |
| `trace_turns_vs_expected` | `actual_turns / expected_actions` — closer to 1.0 = better |
| `trace_guidance_precision` | (Telecom) Fraction of messages with specific device/network terms |

### Domain Routing

| Domain | DAG | Read-After-Write | Guidance |
|---|---|---|---|
| Telecom | 3 DOT files | Yes | Yes |
| Retail | — | Yes | — |
| Airline | — | Yes | — |

</details>

<details>
<summary><strong>Limitations & Future Work</strong> (click to expand)</summary>

- **Telecom-only DAG**: Retail/airline use simpler read-after-write checks. Authoring DAGs for those policies would extend coverage.
- **No LLM-based semantics**: By design. The existing `NLAssertionsEvaluator` pattern could add hallucination detection as a future Layer 2.
- **Adversarial wrapper is illustrative**: Production deployment would need calibration against real user distributions.

</details>

<details>
<summary><strong>Related Work</strong> (click to expand)</summary>

Draws on: TRAJECT-Bench (tool usage diagnostics), TRACE (evidence banks), Agent GPA (temporal phase evaluation), SWE-eval (info-gain metrics). Adapted for tau2-bench's Dec-POMDP dual-control environment with a deterministic-first philosophy.

</details>
