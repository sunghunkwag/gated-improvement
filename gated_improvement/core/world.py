"""Salted task worlds: one fresh, secret world per task.

Adapted from the gated-self-improvement task generator (Apache-2.0). The
generative process is the same; what changes is that every random draw --
including the hidden motif library itself -- is keyed by a secret world key.
Knowing this public source code is therefore not enough to recover a world's
building blocks or its sealed tasks.

A world contains:
  * a hidden MOTIF LIBRARY: 8 "atoms" (2-primitive behaviours no single
    primitive reproduces) and 6 "compounds" (two atoms back to back, whose
    behaviour no program of <= 3 primitives reproduces);
  * 16 task FAMILIES, each = 3 motifs + 2 "glue" primitives; a task is a chain
    of 1-3 motifs from its family with occasional glue between them;
  * a family split: 10 TRAIN families and 6 EXTERNAL families;
  * tasks:  train        (train families)    -- the only split the agent sees
            ext_dev      (external families) -- SEALED, decides the gate
            ext_confirm  (external families) -- SEALED, sets the reward
Every task in a world has a distinct behaviour signature across all splits, so
no sealed task's behaviour appears among the shown tasks.
"""

import functools
import threading

from . import solver as SV
from . import substrate as S

VERSION = "gi1"
N_ATOMS = 8
N_COMPOUNDS = 6
N_FAMILIES = 16
N_TRAIN_FAMILIES = 10
SPLITS = (("train", 40), ("ext_dev", 32), ("ext_confirm", 32))
SEALED_SPLITS = ("ext_dev", "ext_confirm")
MAX_TASK_LEN = 12
UNIT_COUNTS = (1, 1, 2, 2, 3)
GLUE_P = 0.25
_BIG = 10**12


def _sig(prims):
    return S.signature(S.fns_of(prims), S.Meter(_BIG, "task-gen"))


def _nonempty(sig):
    return sum(1 for o in sig if len(o) > 0)


_REF = {}
_REF_LOCK = threading.Lock()


def _reference_signatures():
    """Behaviours of every program of <= 3 primitives. Independent of the
    world key, so computed once per process."""
    with _REF_LOCK:
        if not _REF:
            short = {_sig(())}
            for a in S.NAMES:
                short.add(_sig((a,)))
            upto3 = set(short)
            for a in S.NAMES:
                for b in S.NAMES:
                    upto3.add(_sig((a, b)))
            for a in S.NAMES:
                for b in S.NAMES:
                    for c in S.NAMES:
                        upto3.add(_sig((a, b, c)))
            _REF["short"], _REF["upto3"] = frozenset(short), frozenset(upto3)
    return _REF["short"], _REF["upto3"]


def _motif_library(key):
    short, upto3 = _reference_signatures()
    for attempt in range(32):
        prng = S.XorShift64Star("%s|%s|atoms|%d" % (VERSION, key, attempt))
        atoms, atom_sigs, guard = [], set(), 0
        while len(atoms) < N_ATOMS and guard < 20000:
            guard += 1
            a, b = prng.choice(S.NAMES), prng.choice(S.NAMES)
            if a == b:
                continue
            sg = _sig((a, b))
            if sg in short or sg in atom_sigs or _nonempty(sg) < 5:
                continue
            atoms.append((a, b))
            atom_sigs.add(sg)
        if len(atoms) < N_ATOMS:
            continue
        prng = S.XorShift64Star("%s|%s|compounds|%d" % (VERSION, key, attempt))
        comps, comp_sigs, tries = [], set(), 0
        while len(comps) < N_COMPOUNDS and tries < 20000:
            tries += 1
            i, j = prng.below(N_ATOMS), prng.below(N_ATOMS)
            if i == j:
                continue
            prog = atoms[i] + atoms[j]
            sg = _sig(prog)
            if sg in upto3 or sg in comp_sigs or _nonempty(sg) < 5:
                continue
            comps.append(prog)
            comp_sigs.add(sg)
        if len(comps) == N_COMPOUNDS:
            return atoms, comps
    raise RuntimeError("no motif library could be drawn for this world key")


def _families(key, atoms, comps):
    prng = S.XorShift64Star("%s|%s|families" % (VERSION, key))
    out = []
    for _k in range(N_FAMILIES):
        n_comp = 1 + prng.below(2)
        cs = prng.shuffled(range(N_COMPOUNDS))[:n_comp]
        as_ = prng.shuffled(range(N_ATOMS))[: 3 - n_comp]
        units = [comps[c] for c in cs] + [atoms[a] for a in as_]
        glue = prng.shuffled(S.NAMES)[:2]
        out.append({"units": units, "glue": glue})
    return out


def _sample_program(fam, prng):
    while True:
        n = UNIT_COUNTS[prng.below(len(UNIT_COUNTS))]
        prog = []
        for i in range(n):
            if i > 0 and prng.unit() < GLUE_P:
                prog.append(prng.choice(fam["glue"]))
            prog.extend(prng.choice(fam["units"]))
        if len(prog) <= MAX_TASK_LEN:
            return tuple(prog)


def _inputs(prng, lo, hi, n):
    out = []
    for _ in range(n):
        ln = lo + prng.below(hi - lo + 1)
        out.append([prng.below(256) for _ in range(ln)])
    return out


def _instance(prog, prng):
    m = S.Meter(_BIG, "task-gen")
    fns = S.fns_of(prog)
    tr_in, te_in = _inputs(prng, 6, 9, 4), _inputs(prng, 10, 14, 4)
    tr = [(x, m.run(fns, x)) for x in tr_in]
    te = [(x, m.run(fns, x)) for x in te_in]
    if sum(1 for _x, y in tr if len(y) >= 2) < 3:
        return None  # under-specified: too little output to learn from
    if all(y == x for x, y in tr):
        return None
    if len({tuple(y) for _x, y in tr}) == 1:
        return None
    for nm in S.NAMES:  # reject tasks a single primitive already solves
        f = S.fns_of((nm,))
        if all(m.run(f, x) == y for x, y in tr + te):
            return None
    return tr, te


def _manifest(key, fams):
    prng = S.XorShift64Star("%s|%s|family-split" % (VERSION, key))
    order = prng.shuffled(range(N_FAMILIES))
    train_f = sorted(order[:N_TRAIN_FAMILIES])
    ext_f = sorted(order[N_TRAIN_FAMILIES:])
    seen = set()
    # interleave the splits (k/n order) so no split monopolises the few short
    # behaviours each family has; behaviours stay globally distinct
    sched = sorted(
        (k / float(n), si, split, k) for si, (split, n) in enumerate(SPLITS) for k in range(n)
    )
    rows = {split: [] for split, _n in SPLITS}
    for _frac, _si, split, k in sched:
        pool = train_f if split == "train" else ext_f
        tp = S.XorShift64Star("%s|%s|task|%s|%d" % (VERSION, key, split, k))
        placed = False
        for offset in range(len(pool)):
            fam = fams[pool[(k + offset) % len(pool)]]
            for _try in range(2000):
                prog = _sample_program(fam, tp)
                sg = _sig(prog)
                if sg in seen:
                    continue
                inst = _instance(prog, tp)
                if inst is None:
                    continue
                seen.add(sg)
                rows[split].append({"program": prog, "train": inst[0], "test": inst[1]})
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError("task generation exhausted")
    return rows


class World:
    """Everything one task needs, derived from one secret key."""

    def __init__(self, key):
        self.key = key
        self.atoms, self.compounds = _motif_library(key)
        self.families = _families(key, self.atoms, self.compounds)
        self._rows = _manifest(key, self.families)

    def shown(self):
        """The train-family tasks the agent sees: (train pairs, test pairs)."""
        return [(r["train"], r["test"]) for r in self._rows["train"]]

    def sealed(self, split):
        if split not in SEALED_SPLITS:
            raise KeyError(split)
        return self._rows[split]

    def gold_config(self):
        """Reference improvement for validation: the hidden motif library as
        macros. Never shown to the agent."""
        macros = [SV.macro_token(u) for u in self.atoms + self.compounds]
        cfg = SV.SolverConfig(macros=macros)
        cfg.weights = {t: 1.0 for t in cfg.tokens()}
        return cfg


def world_key(salt, index):
    return S.sha256_text("%s|world|%s|%d" % (VERSION, salt, int(index)))


@functools.lru_cache(maxsize=512)
def world_for(salt, index):
    return World(world_key(salt, index))
