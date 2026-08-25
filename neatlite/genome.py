"""Genome: the evolvable description of one neural network.

Node ids follow the usual NEAT convention:

    inputs  : -1, -2, ... -num_inputs   (no bias, they just hold the sensor value)
    outputs : 0, 1, ... num_outputs - 1
    hidden  : num_outputs, num_outputs + 1, ...  (handed out by InnovationTracker)

Connections are keyed by ``(src, dst)`` which is what actually matters for
alignment during crossover; the innovation number is kept alongside so genomes
can be compared the way the paper describes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NodeGene:
    bias: float

    def copy(self) -> "NodeGene":
        return NodeGene(self.bias)


@dataclass
class ConnGene:
    weight: float
    enabled: bool
    innovation: int

    def copy(self) -> "ConnGene":
        return ConnGene(self.weight, self.enabled, self.innovation)


class InnovationTracker:
    """Hands out stable ids for structural mutations.

    Two genomes that grow the same connection get the same innovation number,
    which is what makes crossover meaningful. The cache is kept for the whole
    run rather than reset every generation -- a common simplification that makes
    identical structures always align.
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._conn_innov: dict[tuple[int, int], int] = {}
        self._node_splits: dict[tuple[int, int], int] = {}
        self._next_innov = 0
        self._next_node = num_outputs

    def conn_innovation(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self._conn_innov:
            self._conn_innov[key] = self._next_innov
            self._next_innov += 1
        return self._conn_innov[key]

    def split_node(self, src: int, dst: int) -> int:
        """Id of the node created by splitting connection ``(src, dst)``."""
        key = (src, dst)
        if key not in self._node_splits:
            self._node_splits[key] = self._next_node
            self._next_node += 1
        return self._node_splits[key]

    def seed_hidden_node(self, index: int) -> int:
        """Id of the ``index``-th hidden node of a freshly seeded genome.

        All initial genomes must agree on these ids, otherwise they would look
        structurally different to each other from generation zero.
        """
        key = ("seed", index)
        if key not in self._node_splits:
            self._node_splits[key] = self._next_node
            self._next_node += 1
        return self._node_splits[key]


def creates_cycle(conns, src: int, dst: int) -> bool:
    """True if adding src -> dst would make the graph non feed-forward."""
    if src == dst:
        return True
    # Walk forward from dst; if we can reach src the new edge closes a loop.
    visited = {dst}
    stack = [dst]
    while stack:
        node = stack.pop()
        for a, b in conns:
            if a == node and b not in visited:
                if b == src:
                    return True
                visited.add(b)
                stack.append(b)
    return False


class Genome:
    __slots__ = ("key", "nodes", "conns", "fitness")

    def __init__(self, key: int):
        self.key = key
        self.nodes: dict[int, NodeGene] = {}
        self.conns: dict[tuple[int, int], ConnGene] = {}
        self.fitness: float = 0.0

    # ------------------------------------------------------------------ init
    @classmethod
    def new(cls, key: int, cfg, tracker: InnovationTracker, rng: random.Random) -> "Genome":
        g = cls(key)
        inputs = [-i - 1 for i in range(cfg.num_inputs)]
        outputs = list(range(cfg.num_outputs))

        for n in outputs:
            g.nodes[n] = NodeGene(rng.gauss(0.0, cfg.bias_init_stdev))

        hidden = []
        for i in range(cfg.num_hidden):
            n = tracker.seed_hidden_node(i)
            hidden.append(n)
            g.nodes[n] = NodeGene(rng.gauss(0.0, cfg.bias_init_stdev))

        if cfg.initial_connection == "none":
            pairs = []
        elif hidden:
            pairs = [(i, h) for i in inputs for h in hidden]
            pairs += [(h, o) for h in hidden for o in outputs]
        else:
            pairs = [(i, o) for i in inputs for o in outputs]

        for src, dst in pairs:
            if cfg.initial_connection == "partial" and rng.random() > cfg.initial_conn_prob:
                continue
            g.add_conn(src, dst, rng.gauss(0.0, cfg.weight_init_stdev), tracker)
        return g

    def copy(self) -> "Genome":
        g = Genome(self.key)
        g.nodes = {k: v.copy() for k, v in self.nodes.items()}
        g.conns = {k: v.copy() for k, v in self.conns.items()}
        g.fitness = self.fitness
        return g

    def add_conn(self, src: int, dst: int, weight: float, tracker: InnovationTracker) -> None:
        self.conns[(src, dst)] = ConnGene(weight, True, tracker.conn_innovation(src, dst))

    # ------------------------------------------------------------- mutation
    def mutate(self, cfg, tracker: InnovationTracker, rng: random.Random) -> None:
        # Structural mutations first, so the fresh genes get a chance to be
        # perturbed in the same round.
        if rng.random() < cfg.node_add_prob:
            self.mutate_add_node(cfg, tracker, rng)
        if rng.random() < cfg.node_delete_prob:
            self.mutate_delete_node(cfg, rng)
        if rng.random() < cfg.conn_add_prob:
            self.mutate_add_conn(cfg, tracker, rng)
        if rng.random() < cfg.conn_delete_prob:
            self.mutate_delete_conn(rng)

        for conn in self.conns.values():
            if rng.random() < cfg.weight_replace_rate:
                conn.weight = rng.gauss(0.0, cfg.weight_init_stdev)
            elif rng.random() < cfg.weight_mutate_rate:
                conn.weight += rng.gauss(0.0, cfg.weight_mutate_power)
            conn.weight = max(-cfg.weight_max, min(cfg.weight_max, conn.weight))
            if rng.random() < cfg.enabled_mutate_rate:
                conn.enabled = not conn.enabled

        for node in self.nodes.values():
            if rng.random() < cfg.bias_replace_rate:
                node.bias = rng.gauss(0.0, cfg.bias_init_stdev)
            elif rng.random() < cfg.bias_mutate_rate:
                node.bias += rng.gauss(0.0, cfg.bias_mutate_power)
            node.bias = max(-cfg.bias_max, min(cfg.bias_max, node.bias))

    def mutate_add_node(self, cfg, tracker: InnovationTracker, rng: random.Random) -> None:
        """Split an existing connection in two, with a new node in the middle."""
        enabled = [k for k, c in self.conns.items() if c.enabled]
        if not enabled:
            return
        key = rng.choice(enabled)
        old = self.conns[key]

        src, dst = key
        new_node = tracker.split_node(src, dst)
        if new_node in self.nodes:  # this edge was already split once
            return
        old.enabled = False
        self.nodes[new_node] = NodeGene(0.0)
        # Weight 1 in, old weight out: the network behaves the same right after
        # the mutation, so the new structure is not immediately punished.
        self.add_conn(src, new_node, 1.0, tracker)
        self.add_conn(new_node, dst, old.weight, tracker)

    def mutate_delete_node(self, cfg, rng: random.Random) -> None:
        hidden = [n for n in self.nodes if n >= cfg.num_outputs]
        if not hidden:
            return
        node = rng.choice(hidden)
        for key in [k for k in self.conns if node in k]:
            del self.conns[key]
        del self.nodes[node]

    def mutate_add_conn(self, cfg, tracker: InnovationTracker, rng: random.Random) -> None:
        inputs = [-i - 1 for i in range(cfg.num_inputs)]
        outputs = list(range(cfg.num_outputs))
        hidden = [n for n in self.nodes if n >= cfg.num_outputs]

        possible_src = inputs + hidden + outputs
        possible_dst = outputs + hidden  # never feed back into an input
        if not possible_dst:
            return

        for _ in range(20):  # a few tries, then give up for this generation
            src = rng.choice(possible_src)
            dst = rng.choice(possible_dst)
            if (src, dst) in self.conns:
                # Re-enabling a dormant link is a cheap way to explore.
                if not self.conns[(src, dst)].enabled:
                    self.conns[(src, dst)].enabled = True
                    return
                continue
            if creates_cycle(self.conns, src, dst):
                continue
            self.add_conn(src, dst, rng.gauss(0.0, cfg.weight_init_stdev), tracker)
            return

    def mutate_delete_conn(self, rng: random.Random) -> None:
        if not self.conns:
            return
        del self.conns[rng.choice(list(self.conns))]

    # ------------------------------------------------------------ crossover
    @staticmethod
    def crossover(parent1: "Genome", parent2: "Genome", key: int, rng: random.Random) -> "Genome":
        """``parent1`` must be the fitter of the two (caller's job)."""
        child = Genome(key)

        for nid, g1 in parent1.nodes.items():
            g2 = parent2.nodes.get(nid)
            if g2 is None:  # disjoint/excess: only the fitter parent contributes
                child.nodes[nid] = g1.copy()
            else:
                child.nodes[nid] = (g1 if rng.random() < 0.5 else g2).copy()

        for ckey, c1 in parent1.conns.items():
            c2 = parent2.conns.get(ckey)
            if c2 is None:
                child.conns[ckey] = c1.copy()
            else:
                gene = (c1 if rng.random() < 0.5 else c2).copy()
                # The paper: a gene disabled in either parent stays disabled 75%
                # of the time.
                if not (c1.enabled and c2.enabled):
                    gene.enabled = rng.random() >= 0.75
                child.conns[ckey] = gene

        return child

    # ----------------------------------------------------------- speciation
    def distance(self, other: "Genome", cfg) -> float:
        """Compatibility distance used to group genomes into species."""
        node_dist = 0.0
        if self.nodes or other.nodes:
            keys1, keys2 = set(self.nodes), set(other.nodes)
            shared = keys1 & keys2
            disjoint = len(keys1 ^ keys2)
            for k in shared:
                node_dist += cfg.compat_weight_coeff * abs(
                    self.nodes[k].bias - other.nodes[k].bias
                )
            node_dist = (node_dist + cfg.compat_disjoint_coeff * disjoint) / max(
                len(keys1), len(keys2)
            )

        conn_dist = 0.0
        if self.conns or other.conns:
            keys1, keys2 = set(self.conns), set(other.conns)
            shared = keys1 & keys2
            disjoint = len(keys1 ^ keys2)
            for k in shared:
                conn_dist += cfg.compat_weight_coeff * abs(
                    self.conns[k].weight - other.conns[k].weight
                )
            conn_dist = (conn_dist + cfg.compat_disjoint_coeff * disjoint) / max(
                len(keys1), len(keys2)
            )

        return node_dist + conn_dist

    def size(self) -> tuple[int, int]:
        return len(self.nodes), sum(1 for c in self.conns.values() if c.enabled)

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "fitness": self.fitness,
            "nodes": {str(k): v.bias for k, v in self.nodes.items()},
            "conns": [
                [src, dst, c.weight, c.enabled, c.innovation]
                for (src, dst), c in self.conns.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        g = cls(data.get("key", 0))
        g.fitness = data.get("fitness", 0.0)
        g.nodes = {int(k): NodeGene(float(v)) for k, v in data["nodes"].items()}
        g.conns = {
            (int(src), int(dst)): ConnGene(float(w), bool(en), int(innov))
            for src, dst, w, en, innov in data["conns"]
        }
        return g

    def __repr__(self) -> str:
        n, c = self.size()
        return f"<Genome {self.key} nodes={n} conns={c} fitness={self.fitness:.1f}>"
