"""Score synthetic replies through verifiers' own Task.score path."""

import asyncio
import json

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

from gated_improvement.core import world as W
from gated_improvement.taskset import GatedImprovementConfig, GatedImprovementTaskset

SALT = "test-suite"


def fenced(obj):
    return "```json\n" + json.dumps(obj) + "\n```"


def make_trace(task, reply):
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=[
            MessageNode(message=UserMessage(content=task.data.prompt_text)),
            MessageNode(parent=0, message=AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_rewards_metrics_and_trace_info():
    task = next(iter(GatedImprovementTaskset(GatedImprovementConfig(num_tasks=1, salt=SALT))))
    w = W.world_for(SALT, 0)
    gold = fenced({"macros": [list(u) for u in w.atoms + w.compounds]})
    for reply, expect_positive in ((gold, True), (fenced({}), False), ("not json", False)):
        trace = make_trace(task, reply)
        assert trace.last_reply == reply
        asyncio.run(task.score(trace))
        assert set(trace.rewards) == {"gated_gain"}
        assert "gate_pass" in trace.metrics and "dev_p_value" in trace.metrics
        json.dumps(trace.info["gate"])
        if expect_positive:
            assert trace.reward > 0.15 and trace.metrics["gate_pass"] == 1.0
        else:
            assert trace.reward == 0.0


def test_salt_comes_from_the_environment_when_unset(monkeypatch):
    import gated_improvement.taskset as T

    monkeypatch.setattr(T, "_PROCESS_SALT", [])
    monkeypatch.setenv(T.SALT_ENV_VAR, "from-env")
    unset = [t.data.prompt_text for t in T.GatedImprovementTaskset(T.GatedImprovementConfig(num_tasks=2))]
    explicit = [
        t.data.prompt_text
        for t in T.GatedImprovementTaskset(T.GatedImprovementConfig(num_tasks=2, salt="from-env"))
    ]
    assert unset == explicit


def test_random_salt_when_nothing_is_set(monkeypatch):
    import gated_improvement.taskset as T

    monkeypatch.setattr(T, "_PROCESS_SALT", [])
    monkeypatch.delenv(T.SALT_ENV_VAR, raising=False)
    first = next(iter(T.GatedImprovementTaskset(T.GatedImprovementConfig(num_tasks=1))))
    monkeypatch.setattr(T, "_PROCESS_SALT", [])
    second = next(iter(T.GatedImprovementTaskset(T.GatedImprovementConfig(num_tasks=1))))
    assert first.data.salt_id != second.data.salt_id


def test_missing_salt_fails_loudly():
    task = next(iter(GatedImprovementTaskset(GatedImprovementConfig(num_tasks=1, salt=SALT))))
    broken = type(task)(task.data, task.config.model_copy(update={"world_salt": ""}))
    try:
        broken._world()
    except RuntimeError as e:
        assert "salt" in str(e)
    else:
        raise AssertionError("a task without a salt must not score")
