from verifiers.v1.harnesses.null import NullHarness

from gated_improvement.taskset import GatedImprovementTaskset


class GatedImprovementHarness(NullHarness):
    """Default harness: a plain chat turn with no tools, matching the
    single-turn task. Pick another harness with `--env.agent.harness.id`."""


__all__ = ["GatedImprovementHarness", "GatedImprovementTaskset"]
