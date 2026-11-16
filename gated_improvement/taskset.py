"""gated-improvement: improve a program-synthesis solver, judged by a sealed gate.

Single turn. The agent sees example tasks (input -> output pairs) from some
task families of a freshly generated world and submits a configuration for a
search-based program synthesizer: new macro tokens, a sampling prior, a length
bound and an exploration rate. It never submits code.

The configuration is scored against the unchanged baseline on sealed tasks
from other families of the same world, with identical search randomness and
an identical execution budget. The reward is non-zero only if a one-sided
sign test over tasks passes on one sealed batch; its value is then the net
share of the baseline's failures fixed on a second, disjoint sealed batch.
See `core/gate.py` for the full rule and `scripts/exploit_report.py` for the
reward-hacking strategies it is tested against.

Every task has its own world, drawn from a secret salt, so neither the hidden
building blocks nor the sealed tasks can be recovered from this source code.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import threading

import verifiers.v1 as vf

from gated_improvement.core import gate as G
from gated_improvement.core import solver as SV
from gated_improvement.core import substrate as S
from gated_improvement.core import world as W

PRIMITIVE_DOCS = (
    ("inc", "add 1 to every element"),
    ("dec", "subtract 1 from every element"),
    ("double", "multiply every element by 2"),
    ("half", "floor-divide every element by 2"),
    ("neg", "replace every element x by 255 - x"),
    ("add5", "add 5 to every element"),
    ("mod10", "replace every element x by x mod 10"),
    ("reverse", "reverse the list"),
    ("sort", "sort ascending"),
    ("rotate", "move the first element to the end"),
    ("dedupe", "drop every element equal to the one before it"),
    ("evens", "keep only even elements"),
    ("big", "keep only elements greater than 128"),
    ("tail", "drop the first element"),
    ("init", "drop the last element"),
    ("cumsum", "running sums"),
    ("cummax", "running maxima"),
    ("diff", "x[i] - x[i-1] for every i > 0; the first element is kept"),
    ("dup", "repeat every element twice"),
    ("pairsum", "x[i] + x[i+1] for each adjacent pair (one element shorter)"),
)
assert tuple(sorted(n for n, _ in PRIMITIVE_DOCS)) == S.NAMES

_SYSTEM_TEMPLATE = (
    """You improve a program-synthesis solver. You cannot change its code, only its configuration, which is data.

THE DOMAIN
A program maps a list of integers (0-255) to a list of integers. It is a pipeline: a sequence of primitives applied left to right. After every primitive each value is taken modulo 256 and the list is cut to its first 24 elements. The 20 primitives:
"""
    + "\n".join("  %-8s %s" % (n, d) for n, d in PRIMITIVE_DOCS)
    + """

Every task is a pipeline built by chaining a few hidden building blocks -- short fixed sequences of primitives -- sometimes with one extra primitive between them.

THE SOLVER
For each task the solver runs an evolutionary search over programs written as token sequences. It sees only the task's example pairs and may execute programs at most {budget:,} times. A task counts as solved only if the program it finds is exact on the examples and on hidden longer test inputs. Tokens are the 20 primitives plus any macros you define. A program has at most max_tokens tokens and at most 16 primitives once macros are expanded.

YOUR CONFIGURATION
  macros      list of new tokens, each a list of 2 to 8 primitive names, e.g. ["sort", "diff"]; at most 32
  weights     sampling prior per token: keys are primitive names or macros written "sort.diff"; values 0 to 1000; unlisted tokens get 1.0
  max_tokens  integer 1 to 8
  p_random    number 0 to 1: the share of brand-new random programs in each generation
The baseline has no macros, every weight 1.0, max_tokens 4 and p_random 0.3. Any key you leave out keeps its baseline value.

HOW YOU ARE SCORED
The tasks below come from some families. Your configuration is tested on hidden tasks from other families, which reuse the same hidden building blocks in new combinations. Your configuration and the baseline search the same hidden tasks with identical randomness and the same execution budget, so searching longer cannot help; only a better prior can. If, on one batch of hidden tasks, more tasks improve than get worse and a one-sided sign test over tasks is significant at the {alpha:g} level, your reward is the net share of the baseline's failures that you fix on a second, separate batch of hidden tasks (negative if you break more than you fix). Otherwise your reward is 0. Resubmitting the baseline scores 0. Macros that are not real building blocks dilute the prior and usually make the solver worse.

ANSWER FORMAT
You may reason first. End your reply with exactly one JSON object in a ```json code block, using only the keys macros, weights, max_tokens and p_random. An unknown key, unknown primitive or out-of-range value makes the answer invalid, which scores 0."""
)


def system_prompt(budget: int = 3000, alpha: float = 0.05) -> str:
    return _SYSTEM_TEMPLATE.replace("{budget:,}", "{:,}".format(budget)).replace(
        "{alpha:g}", "{:g}".format(alpha)
    )


def _fmt(xs):
    return "[" + ", ".join(str(v) for v in xs) + "]"


def build_prompt(world: W.World, n_shown: int) -> str:
    lines = [
        "Example tasks from the training families. Each line is input -> output; "
        "every task has eight pairs, the last four on longer inputs.",
        "",
    ]
    for i, (train, test) in enumerate(world.shown()[:n_shown], 1):
        lines.append("Task %d" % i)
        for x, y in list(train) + list(test):
            lines.append("  %s -> %s" % (_fmt(x), _fmt(y)))
        lines.append("")
    lines.append(
        "Propose a configuration that helps the solver on hidden tasks built from the same building blocks."
    )
    return "\n".join(lines)


SALT_ENV_VAR = "GATED_IMPROVEMENT_SALT"

_PROCESS_SALT: list[str] = []
_SALT_LOCK = threading.Lock()


def _process_salt() -> str:
    """The salt used when the config leaves `salt` unset: $GATED_IMPROVEMENT_SALT
    if it is set (every worker of a multi-process run then builds the same worlds,
    and the secret stays out of run configs and command lines), else a random
    salt fixed for the life of this process."""
    with _SALT_LOCK:
        if not _PROCESS_SALT:
            _PROCESS_SALT.append(os.environ.get(SALT_ENV_VAR) or secrets.token_hex(16))
        return _PROCESS_SALT[0]


class GatedImprovementData(vf.TaskData):
    world_index: int
    """Which world of the taskset's salt this task uses. The salt itself lives in
    the evaluator-side task config, never in task data."""
    salt_id: str
    """Non-secret fingerprint of the salt, for auditing traces."""


class GatedImprovementTaskConfig(vf.TaskConfig):
    world_salt: str = ""
    """Filled in by the taskset from its `salt`; do not set it directly."""
    n_streams: int = 2
    """Search random streams per sealed task, for each of candidate and baseline."""
    alpha: float = 0.05
    """One-sided sign-test level for the gate."""
    eval_budget: int = 3000
    """Program executions allowed per sealed task search (identical for both arms)."""
    max_turns: int = 1
    """Model turns before the rollout stops. 1 is the single-turn reasoning task; raise
    it with a code-executing harness for the agentic variant (use an isolated runtime)."""


class GatedImprovementTask(
    vf.Task[GatedImprovementData, vf.State, GatedImprovementTaskConfig]
):
    @vf.stop
    async def turn_limit(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= self.config.max_turns

    def _settings(self) -> G.GateSettings:
        return G.GateSettings(
            n_streams=self.config.n_streams,
            alpha=self.config.alpha,
            budget=self.config.eval_budget,
        )

    def _world(self) -> W.World:
        if not self.config.world_salt:
            raise RuntimeError(
                "gated-improvement: the task has no world salt; build tasks through "
                "GatedImprovementTaskset, and set `salt` explicitly when rescoring"
            )
        return W.world_for(self.config.world_salt, self.data.world_index)

    async def _gate(self, trace: vf.Trace) -> dict:
        result = trace.info.get("gate")
        if result is None:
            result = await asyncio.to_thread(
                G.score_reply, self._world(), trace.last_reply or "", self._settings()
            )
            trace.info["gate"] = result
        return result

    @vf.metric
    async def gate_report(self, trace: vf.Trace) -> dict[str, float]:
        result = await self._gate(trace)
        return {k: float(v) for k, v in result.items() if k not in ("reward", "error")}

    @vf.reward(weight=1.0)
    async def gated_gain(self, trace: vf.Trace) -> float:
        return float((await self._gate(trace))["reward"])

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The hidden motif library as macros must pass the gate with a positive
        reward, and the baseline must score exactly zero."""
        world = await asyncio.to_thread(self._world)
        gold = await asyncio.to_thread(G.score_config, world, world.gold_config(), self._settings())
        noop = await asyncio.to_thread(G.score_config, world, SV.base_config(), self._settings())
        return gold["gate_pass"] == 1.0 and gold["reward"] > 0.0 and noop["reward"] == 0.0


class GatedImprovementConfig(vf.TasksetConfig):
    num_tasks: int = 512
    """Number of tasks; each has its own freshly generated world."""
    salt: str | None = None
    """Secret that generates the worlds. Unset: $GATED_IMPROVEMENT_SALT if set, else a
    random salt per process, so worlds never repeat across runs. Set it (or the
    variable) to reproduce a run, to rescore its traces, or to give every worker of
    a multi-process run the same worlds."""
    n_shown: int = 40
    """How many training-family tasks to show in the prompt (at most 40)."""
    task: GatedImprovementTaskConfig = GatedImprovementTaskConfig()


class GatedImprovementTaskset(vf.Taskset[GatedImprovementTask, GatedImprovementConfig]):
    def load(self):
        salt = self.config.salt or _process_salt()
        task_config = self.config.task.model_copy(update={"world_salt": salt})
        salt_id = S.sha256_text("salt-id|" + salt)[:10]
        prompt_system = system_prompt(task_config.eval_budget, task_config.alpha)
        for i in range(self.config.num_tasks):
            world = W.world_for(salt, i)
            yield GatedImprovementTask(
                GatedImprovementData(
                    id="gi-%s-%d" % (salt_id, i),
                    name="world-%d" % i,
                    prompt=build_prompt(world, self.config.n_shown),
                    system_prompt=prompt_system,
                    world_index=i,
                    salt_id=salt_id,
                ),
                task_config,
            )
