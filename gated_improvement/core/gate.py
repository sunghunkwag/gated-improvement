"""The reward: a sealed, compute-matched, sign-test gate with a fresh confirm.

Scoring a candidate configuration C against the unchanged baseline B:

 1. Parse strictly. Anything outside the schema (unknown key, unknown
    primitive, non-finite or out-of-range number, oversized answer) is
    invalid and scores 0.
 2. A configuration that behaves exactly like the baseline (same macros
    after dropping zero-weight ones, same normalised weights, same
    max_tokens and p_random) is a no-op and scores 0.
 3. GATE (ext_dev, sealed): C and B each search every ext_dev task with the
    same execution budget and the same random stream -- streams are keyed
    by (world, split, task, stream index), never by the configuration. For
    each task, d = mean over streams of (C solved - B solved). The gate
    passes only if more tasks improve than regress and a one-sided exact
    sign test over tasks gives p <= alpha.
 4. CONFIRM (ext_confirm, sealed, disjoint tasks): only for gated
    candidates, reward = (C solved - B solved) / (tasks B fails), averaged
    over streams and clipped to [-1, 1]. It is measured on tasks that played
    no part in the gate decision, so a candidate that passed the gate by
    luck has zero expected reward rather than a positive one.

Everything is deterministic given the world key and the candidate.
"""

import json
import math
import re
import threading
from dataclasses import dataclass

from . import solver as SV
from . import substrate as S
from .world import VERSION, World

MAX_JSON_CHARS = 20_000
MAX_MACROS = 32
MIN_MACRO_LEN = 2
MAX_MACRO_LEN = 8
MAX_WEIGHT = 1000.0
MIN_TOKENS, MAX_TOKENS = 1, 8
ALLOWED_KEYS = frozenset({"macros", "weights", "max_tokens", "p_random"})

_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class GateSettings:
    n_streams: int = 2
    alpha: float = 0.05
    budget: int = 3000


class InvalidAnswer(ValueError):
    pass


# --------------------------------------------------------------------------
# Parsing


def extract_json(reply):
    """The last fenced code block, or the whole reply if it has none."""
    if not reply or not reply.strip():
        raise InvalidAnswer("empty reply")
    blocks = _FENCE.findall(reply)
    raw = blocks[-1].strip() if blocks else reply.strip()
    if len(raw) > MAX_JSON_CHARS:
        raise InvalidAnswer("answer exceeds %d characters" % MAX_JSON_CHARS)
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError):
        raise InvalidAnswer("answer is not valid JSON") from None
    if not isinstance(obj, dict):
        raise InvalidAnswer("answer is not a JSON object")
    return obj


def _number(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidAnswer("%s must be a number" % what)
    v = float(v)
    if not math.isfinite(v):
        raise InvalidAnswer("%s must be finite" % what)
    return v


def _macro(m):
    if isinstance(m, str):
        body = m.strip()
        if body.startswith("<") and body.endswith(">"):
            body = body[1:-1]
        prims = tuple(p.strip() for p in body.split("."))
    elif isinstance(m, list) and all(isinstance(p, str) for p in m):
        prims = tuple(m)
    else:
        raise InvalidAnswer("each macro must be a list of primitive names")
    if not (MIN_MACRO_LEN <= len(prims) <= MAX_MACRO_LEN):
        raise InvalidAnswer("macros need %d-%d primitives" % (MIN_MACRO_LEN, MAX_MACRO_LEN))
    for p in prims:
        if p not in S.PRIMS:
            raise InvalidAnswer("unknown primitive %r" % (p,))
    return prims


def build_config(obj):
    """Validate a parsed answer and turn it into a SolverConfig."""
    extra = sorted(set(obj) - ALLOWED_KEYS)
    if extra:
        raise InvalidAnswer("unknown key(s): %s" % ", ".join(map(str, extra)))

    raw_macros = obj.get("macros", [])
    if not isinstance(raw_macros, list):
        raise InvalidAnswer("macros must be a list")
    if len(raw_macros) > MAX_MACROS:
        raise InvalidAnswer("at most %d macros" % MAX_MACROS)
    macros = []
    for m in raw_macros:
        tok = SV.macro_token(_macro(m))
        if tok not in macros:
            macros.append(tok)

    tokens = set(S.NAMES) | set(macros)
    raw_w = obj.get("weights", {})
    if not isinstance(raw_w, dict):
        raise InvalidAnswer("weights must be an object")
    weights = {t: 1.0 for t in tokens}
    for k, v in raw_w.items():
        name = k.strip()
        tok = name if name in S.PRIMS else SV.macro_token(_macro(name))
        if tok not in tokens:
            raise InvalidAnswer("weight for undeclared macro %r" % (k,))
        w = _number(v, "weight for %r" % (k,))
        if not (0.0 <= w <= MAX_WEIGHT):
            raise InvalidAnswer("weights must be between 0 and %g" % MAX_WEIGHT)
        weights[tok] = w
    if sum(weights.values()) <= 0.0:
        raise InvalidAnswer("weights must not all be zero")

    mt = obj.get("max_tokens", SV.BASE_MAX_TOKENS)
    if isinstance(mt, bool) or not isinstance(mt, int):
        raise InvalidAnswer("max_tokens must be an integer")
    if not (MIN_TOKENS <= mt <= MAX_TOKENS):
        raise InvalidAnswer("max_tokens must be %d-%d" % (MIN_TOKENS, MAX_TOKENS))

    pr = _number(obj.get("p_random", SV.BASE_P_RANDOM), "p_random")
    if not (0.0 <= pr <= 1.0):
        raise InvalidAnswer("p_random must be between 0 and 1")

    return SV.SolverConfig(macros=macros, weights=weights, max_tokens=mt, p_random=pr)


def behaviour_key(cfg):
    """Configurations with equal keys search identically: zero-weight macros
    are never sampled, and only weight ratios matter."""
    live = [m for m in cfg.macros if cfg.weights.get(m, 0.0) > 0.0]
    toks = tuple(S.NAMES) + tuple(live)
    total = sum(max(cfg.weights.get(t, 0.0), 0.0) for t in toks)
    weights = {t: round(max(cfg.weights.get(t, 0.0), 0.0) / total, 9) for t in toks}
    return S.canon(
        {
            "macros": list(live),
            "weights": weights,
            "max_tokens": cfg.max_tokens,
            "p_random": round(cfg.p_random, 9),
        }
    )


# --------------------------------------------------------------------------
# Sealed, metered, common-random-number evaluation


def eval_stream(world_key, split, k, stream):
    """Search randomness for one sealed task: keyed by world, split, task and
    stream -- never by the configuration under test."""
    return S.XorShift64Star("%s|%s|eval|%s|%d|%d" % (VERSION, world_key, split, k, stream))


def run_split(world, cfg, split, stream, budget):
    """Per-task solved flags (0/1) for one sealed split and one stream."""
    rows = world.sealed(split)  # fetched before counting: generation is not search
    comp = SV.Compiled(cfg)
    g0 = S.exec_count()
    search_m = S.Meter(10**12, "eval-search")
    score_m = S.Meter(10**12, "eval-score")
    flags = []
    for k, row in enumerate(rows):
        m = search_m.child(budget, "eval-task")
        res = SV.search(comp, row["train"], budget, eval_stream(world.key, split, k, stream), m)
        if res.spent > budget:
            raise S.HiddenComputeError("search overspent its budget")
        ok = res.program is not None and SV.check(comp, res.program, row["train"] + row["test"], score_m)
        flags.append(1 if ok else 0)
    if S.exec_count() - g0 != search_m.spent + score_m.spent:
        raise S.HiddenComputeError("evaluator executed unmetered programs")
    return tuple(flags)


_BASE_CACHE = {}
_BASE_LOCK = threading.Lock()


def baseline_flags(world, split, stream, budget):
    key = (world.key, split, stream, budget)
    with _BASE_LOCK:
        hit = _BASE_CACHE.get(key)
    if hit is not None:
        return hit
    flags = run_split(world, SV.base_config(), split, stream, budget)
    with _BASE_LOCK:
        if len(_BASE_CACHE) > 4096:
            _BASE_CACHE.clear()
        _BASE_CACHE[key] = flags
    return flags


def sign_test_p(n_up, n_down):
    """One-sided exact sign test: P(X >= n_up), X ~ Binomial(n_up + n_down, 1/2)."""
    n = n_up + n_down
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(n_up, n + 1)) / float(2**n)


def _result(**kw):
    out = {
        "reward": 0.0,
        "valid": 0.0,
        "nop": 0.0,
        "gate_pass": 0.0,
        "dev_tasks_up": 0.0,
        "dev_tasks_down": 0.0,
        "dev_p_value": 1.0,
        "confirm_baseline_solved": 0.0,
        "confirm_candidate_solved": 0.0,
        "confirm_gain": 0.0,
        "n_macros": 0.0,
        "error": None,
    }
    out.update(kw)
    return out


def score_config(world, cfg, settings=GateSettings()):
    """Gate one validated configuration in one world. Returns a JSON-able dict."""
    if not isinstance(world, World):
        raise TypeError("world must be a World")
    n_macros = float(len(cfg.macros))
    if behaviour_key(cfg) == behaviour_key(SV.base_config()):
        return _result(valid=1.0, nop=1.0, n_macros=n_macros)

    streams = range(settings.n_streams)
    n_dev = len(world.sealed("ext_dev"))
    diff = [0.0] * n_dev
    for s in streams:
        b = baseline_flags(world, "ext_dev", s, settings.budget)
        c = run_split(world, cfg, "ext_dev", s, settings.budget)
        for k in range(n_dev):
            diff[k] += (c[k] - b[k]) / float(settings.n_streams)
    n_up = sum(1 for d in diff if d > 0)
    n_down = sum(1 for d in diff if d < 0)
    p = sign_test_p(n_up, n_down)
    gate = n_up > n_down and p <= settings.alpha
    out = _result(
        valid=1.0,
        n_macros=n_macros,
        dev_tasks_up=float(n_up),
        dev_tasks_down=float(n_down),
        dev_p_value=p,
        gate_pass=1.0 if gate else 0.0,
    )
    if not gate:
        return out

    n_conf = len(world.sealed("ext_confirm"))
    b_sum = c_sum = 0.0
    for s in streams:
        b_sum += sum(baseline_flags(world, "ext_confirm", s, settings.budget))
        c_sum += sum(run_split(world, cfg, "ext_confirm", s, settings.budget))
    b_mean = b_sum / settings.n_streams
    c_mean = c_sum / settings.n_streams
    headroom = n_conf - b_mean
    ratio = (c_mean - b_mean) / headroom if headroom > 0 else 0.0
    out.update(
        confirm_baseline_solved=b_mean,
        confirm_candidate_solved=c_mean,
        confirm_gain=c_mean - b_mean,
        reward=max(-1.0, min(1.0, ratio)),
    )
    return out


def score_reply(world, reply, settings=GateSettings()):
    """Parse an agent's reply and gate it. Never raises on bad answers."""
    try:
        cfg = build_config(extract_json(reply))
    except InvalidAnswer as e:
        return _result(error=str(e))
    return score_config(world, cfg, settings)
