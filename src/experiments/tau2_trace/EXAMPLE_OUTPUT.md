# Case Studies: Real-World tau2-TRACE Analysis

> From actual tau2-bench simulations (`gpt-4.1-mini`, Sierra orchestrator). Not synthetic.

---

## Case 1: Telecom — "The Expensive Pass"

**Task**: Mobile data issue (airplane mode on + roaming disabled)
**tau2-bench**: reward=1.0, all assertions pass. **A perfect score.**

tau2-TRACE shows the cost behind that score:

| Finding | Value | Why It Matters |
|---|---|---|
| **28 turns** for a 2-action task | `turns_vs_expected = 14.0x` | At scale, this agent erodes margins under outcomes-based pricing |
| **13 total tool calls** (7 agent, 6 user) | vs 2 expected actions | Heavy API consumption for a simple fix |
| **1 redundant call** detected | `check_status_bar` called twice with same args | Wasted compute — deterministically flagged |
| **Workflow: `path2_mobile_data`** matched | From existing DOT files | Agent chose the correct troubleshooting path |
| **87.5% guidance precision** | 7 of 8 messages used specific terms | "Toggle airplane mode", "check roaming" — actionable instructions |
| **$0.018 cost** | vs $0.006 for the failing retail agent | 3x more expensive despite both having zero process errors |

**Developer action**: Use `trace_turns_vs_expected` as a regression metric. This agent works but needs efficiency optimization before production deployment.

---

## Case 2: Retail — "The Process-Perfect Failure"

**Task**: Order item exchange
**tau2-bench**: reward=0.0, exchange action failed. **A total failure.**

tau2-TRACE shows the process was actually sound:

| Finding | Value | Why It Matters |
|---|---|---|
| **0 redundant calls** | Clean tool discipline | No wasted API calls |
| **0 loops, 0 errors** | Flawless execution path | The agent's reasoning was correct |
| **Correct tool sequence** | find_user → get_order → get_product ×2 | Policy adherence = 1.0 |
| **2.6x turn overhead** | 13 turns vs 5 expected actions | Within acceptable range |
| **$0.006 cost** | 3x cheaper than the passing telecom agent | More cost-efficient |
| **Failure isolated** to `exchange_delivered_order_items` | Everything before it passed | Likely a parameter issue, not a reasoning breakdown |

**Developer action**: Fix the exchange tool parameters. Don't redesign the agent's planning — the process is already production-quality. This diagnosis narrows debugging scope from "why did the agent fail?" to "why did this one tool call fail?"

---

## The Point

| | Telecom (PASS) | Retail (FAIL) |
|---|---|---|
| **tau2-bench** | 1.0 | 0.0 |
| **Turn overhead** | 14.0x | 2.6x |
| **Cost** | $0.018 | $0.006 |
| **Process errors** | 0 | 0 |
| **Redundancy** | 1 | 0 |

`pass^k` says one agent is perfect and the other is worthless. tau2-TRACE says the "failing" agent has better process quality, costs less, and is one parameter fix away from production. **That's the diagnostic gap this experiment fills.**

---

## Reproduction

```bash
cd tau2-bench
uv venv .venv --python 3.12 && uv pip install -e "."

# Tests (no API key)
.venv/bin/python -m pytest src/experiments/tau2_trace/tests/ -v

# Real simulation (requires OPENAI_API_KEY)
.venv/bin/tau2 run --domain telecom --num-tasks 1 --num-trials 1 \
    --agent-llm gpt-4.1-mini --user-llm gpt-4.1-mini --save-to my_run

# Analysis
.venv/bin/python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json
```

Pre-computed results from these runs are in `results/`.
