# gated-improvement

A single-turn RL environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) in which the agent improves a program-synthesis solver and is rewarded only for improvement that survives a sealed, compute-matched, statistically gated holdout.

The agent never writes code and never sees the tasks it is graded on. It reads example tasks, infers the hidden building blocks they are made of, and submits a configuration for a search-based solver: new macro tokens, a sampling prior, a length bound and an exploration rate. The configuration is then raced against the unchanged baseline on hidden tasks from families it was never shown, with the same random streams and the same execution budget. The design goal is that the cheapest way to raise the reward is to actually make the solver better.

## The task

Each task has its own world, generated from a secret salt:

- **Domain.** Programs are pipelines of 20 list primitives (`sort`, `diff`, `cumsum`, `dup`, ...) on lists of bytes. The system prompt documents every primitive exactly.
- **Hidden library.** Each world draws 8 two-primitive blocks ("atoms") and 6 compounds of two atoms. Each family uses three of these units and two "glue" primitives; a task chains one to three of its family's units, sometimes with a glue primitive between two of them, up to 12 primitives in all.
- **Families and splits.** 16 families per world: 10 are shown, 6 are held out. The prompt shows 40 tasks from the shown families, each with 8 input/output pairs (4 on longer inputs). Grading uses 32 + 32 tasks from the held-out families: `ext_dev` decides the gate, `ext_confirm` sets the reward. Every task in a world has a distinct behaviour, so no sealed task repeats a shown one.
- **Solver.** An evolutionary search over token sequences with at most 3,000 program executions per task. A task is solved only if the program found is exact on the examples and on longer hidden test inputs. The baseline has no macros, uniform weights, `max_tokens` 4 and `p_random` 0.3.
- **Answer.** One JSON object in a fenced block, with optional keys `macros`, `weights`, `max_tokens`, `p_random`. Example:

```json
{"macros": [["sort", "diff"], ["dup", "pairsum"]], "weights": {"sort.diff": 3}, "max_tokens": 4}
```

## How it is scored

1. **Strict parsing.** The last fenced block (or the whole reply, if it has none) must be a JSON object. An unknown key, unknown primitive, non-finite or out-of-range number, more than 32 macros or an answer longer than 20,000 characters makes it invalid: reward 0.
2. **No-op detection.** A configuration that searches exactly like the baseline (after dropping zero-weight macros and normalising weights) scores 0.
3. **Gate on `ext_dev`.** Candidate and baseline search every sealed task with the same budget and the same random streams, keyed by (world, split, task, stream) and never by the configuration. Per task, the solved difference is averaged over 2 streams. The gate passes only if more tasks improve than regress and a one-sided exact sign test over tasks gives p ≤ 0.05.
4. **Reward on `ext_confirm`.** Only a gated candidate is measured on 32 further sealed tasks that played no part in the gate decision. Reward = (candidate solved − baseline solved) / (tasks the baseline fails), averaged over streams and clipped to [−1, 1]. A candidate that passed the gate by luck therefore has an expected reward of zero, not a positive one, and a candidate that breaks more than it fixes is penalised.

Every execution inside the evaluator is metered; a search that overspends its budget, or any program run that no meter paid for, raises an error instead of producing a score. Scoring is deterministic given the salt and the answer.

## Attacks it was tested against

`scripts/exploit_report.py` runs each strategy below in fresh worlds and reports gate passes and rewards. Output of `python scripts/exploit_report.py --worlds 20 --salt report-final`:

| Strategy | Kind | Submissions | Valid | Gate passed | Mean reward | Max reward |
|---|---|---:|---:|---:|---:|---:|
| Resubmit the baseline | exploit | 20 | 20 | 0 | +0.000 | +0.000 |
| Scale every weight by 7 | exploit | 20 | 20 | 0 | +0.000 | +0.000 |
| True library as macros, all at weight 0 | exploit | 20 | 20 | 0 | +0.000 | +0.000 |
| Tiny weight jitter (near-pure noise) | exploit | 120 | 120 | 0 | +0.000 | +0.000 |
| Random weights (noise) | exploit | 80 | 80 | 0 | +0.000 | +0.000 |
| 14 random macros (noise) | exploit | 40 | 40 | 0 | +0.000 | +0.000 |
| 32 guessed primitive pairs (shotgun) | exploit | 20 | 20 | 0 | +0.000 | +0.000 |
| Longer programs, more random restarts | exploit | 20 | 20 | 0 | +0.000 | +0.000 |
| Search settings only (5 fixed variants) | exploit | 100 | 100 | 0 | +0.000 | +0.000 |
| 32 most frequent atoms of 600 self-salted worlds | exploit | 20 | 20 | 0 | +0.000 | +0.000 |
| Library + sealed programs of a wrongly-salted world | exploit | 20 | 20 | 0 | +0.000 | +0.000 |
| Malformed or injected answer | exploit | 200 | 20 | 0 | +0.000 | +0.000 |
| True motif library as macros | real | 20 | 20 | 20 | +0.434 | +0.647 |
| All 8 atoms (partial knowledge) | real | 20 | 20 | 19 | +0.223 | +0.412 |
| 4 atoms (partial knowledge) | real | 20 | 20 | 5 | +0.026 | +0.140 |
| 4 atoms padded with 10 random macros | real | 20 | 20 | 3 | +0.016 | +0.121 |

The "real" rows set the scale of the reward. The hidden library itself, the answer the environment is built to reward, averages 0.43 rather than 1 because the solver's budget still binds, so rewards compare models with each other rather than against an absolute ceiling. Knowing half the atoms passes the gate only some of the time at the default `alpha`, and padding real blocks with guesses lowers the reward. The report takes about eight minutes on two cores.

The same strategies, plus prompt-leak, determinism, compute-integrity and common-random-number checks, are in `tests/` (`pytest tests`, about two minutes).

## Calibration

The same 10 further worlds and the same answers at three gate levels (one gate run per answer; the decision at each level follows from its p-value). Each cell is gate passes, then mean reward.

| Answer | Submissions | alpha 0.05 | alpha 0.1 | alpha 0.2 |
|---|---:|---:|---:|---:|
| Tiny random weight changes (no information) | 80 | 0, +0.000 | 1, −0.000 | 7, −0.001 |
| 2 true atoms | 10 | 2, +0.011 | 4, +0.024 | 7, +0.020 |
| 4 true atoms | 10 | 5, +0.049 | 6, +0.060 | 6, +0.060 |
| All 8 atoms | 10 | 9, +0.232 | 9, +0.232 | 10, +0.252 |
| Atoms and compounds (full library) | 10 | 10, +0.469 | 10, +0.469 | 10, +0.469 |

Noise passes less often than the nominal level because the exact sign test is conservative at these sample sizes, and when it does pass, the separate confirm batch keeps its reward at zero on average. `alpha` is the density knob: a looser gate rewards partial knowledge more often, at the cost of more zero-mean noise in the reward.

## First model run

Claude Opus 5.5 answered four fresh worlds (single turn, `n_shown` 20, default gate) from the prompt alone, without tools. It is the agent that built this environment, so it knew the generator's design but not the secret salt or any hidden library; treat this as a first difficulty probe, not an independent benchmark. Prompts, answers, the salt (published so the scores can be re-checked; never reuse it for evaluation) and scores are in `results/2026-11-16-claude-opus-5.5/`.

| World | Macros proposed | True blocks among them | Sealed tasks up / down | Sign-test p | Reward | Hidden library's reward |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 1 | 6 / 1 | 0.063 | 0 | 0.426 |
| 1 | 6 | 1 | 1 / 4 | 0.969 | 0 | 0.552 |
| 2 | 6 | 3 | 5 / 2 | 0.227 | 0 | 0.577 |
| 3 | 5 | 2 | 3 / 2 | 0.500 | 0 | 0.347 |

The model recovered 1–3 of the hidden blocks per world and moved more sealed tasks up than down overall (15 vs 9), but never enough for the gate, so every reward is 0; world 0 missed it at p = 0.063. At this size the task is hard and its reward sparse for a strong model answering in one turn, which is the regime where `alpha` (0.1 would have passed world 0) and `n_shown` matter. Results for other models are welcome.

## Quickstart

```bash
# From the Environments Hub
prime env install intelligence-research-project/gated-improvement
# or from source: git clone https://github.com/sunghunkwag/gated-improvement && pip install -e gated-improvement

# Gold check (the hidden library as macros passes the gate) and no-op check (the baseline scores 0)
vf-validate gated-improvement -n 3 --runtime.type subprocess

# Evaluate a model; the default harness is a single chat turn with no tools
vf-run gated-improvement -m <model> -n 32 -r 4 --env.agent.runtime.type subprocess
```

`scripts/mock_model_server.py` is a stand-in OpenAI-compatible model with a scripted policy (true library, baseline, prose) for exercising a full rollout without an API key.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `num_tasks` | 512 | Tasks, one fresh world each |
| `salt` | unset | Secret that generates the worlds. Unset: `$GATED_IMPROVEMENT_SALT` if set, else random per process, so worlds never repeat across runs. Set it to reproduce or rescore a run; give every worker of a multi-process run the same value, preferably through the environment variable so it stays out of run configs |
| `n_shown` | 40 | Example tasks in the prompt (40 ≈ 17–24k tokens, 24 ≈ 11–14k, 12 ≈ 5–7k) |
| `task.alpha` | 0.05 | Gate level (see Calibration) |
| `task.n_streams` | 2 | Random streams per sealed task and arm |
| `task.eval_budget` | 3000 | Program executions per sealed search, identical for both arms |
| `task.max_turns` | 1 | Turns before the rollout stops |

The reward is `gated_gain`. Recorded metrics: `valid`, `nop`, `gate_pass`, `dev_tasks_up`, `dev_tasks_down`, `dev_p_value`, `confirm_baseline_solved`, `confirm_candidate_solved`, `confirm_gain`, `n_macros`. The full gate result, including the parse error of an invalid answer, is kept in `trace.info["gate"]`.

Scoring runs on the evaluator's CPU: roughly 1–6 seconds of one core per rollout, the larger figure for answers that pass the gate and for the first rollout in a world, which also computes and caches the baseline.

## Limitations

- **Synthetic domain.** The skill measured is inducing reusable building blocks from input/output examples and encoding them as a search prior. That is one ingredient of self-improvement, not self-improvement in general.
- **Sparse for weak answers.** Partial knowledge is rewarded only when it moves enough sealed tasks to pass the gate. Early in training most rollouts may score 0; `alpha`, `n_shown` and the number of streams are the levers.
- **Secrecy is the salt.** The generator is public; anyone with the salt can recompute every world. Never evaluate on a salt that has been published.
- **Metering is an integrity check, not a sandbox.** It catches unmetered execution inside the evaluator. In the default single-turn setup the agent runs no code. A multi-turn variant with a code-executing harness must use an isolated runtime (`docker` or `prime`), not `subprocess`.
- **One model so far.** Only the first model run above exists, by the agent that built the environment. Independent baselines are still missing.

## Provenance and license

Derived from [gated-self-improvement](https://github.com/sunghunkwag/gated-self-improvement), a recursive self-improvement experiment whose gated acceptance rule (NOP control, sealed evaluation, rollback) this environment turns into a reward. Licensed under the Apache License, Version 2.0; see `LICENSE` and `NOTICE`.

© 2026 Intelligence Research Project
