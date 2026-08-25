"""Turning a genome into something that can be evaluated.

A genome is a bag of genes; this compiles it once into a topologically ordered
list of ``(node, bias, [(src, weight), ...])`` so that ``activate`` is a couple
of tight loops.
"""

from __future__ import annotations

import math


def required_nodes(inputs, outputs, conns):
    """Nodes that actually take part in producing an output.

    Hidden nodes that no output depends on are dead weight; skipping them keeps
    activation cheap even when mutation has left junk lying around.
    """
    required = set(outputs)
    while True:
        # Anything feeding a required node is itself required.
        new = {a for a, b in conns if b in required and a not in required}
        new -= set(inputs)
        if not new:
            return required
        required |= new


def feed_forward_layers(inputs, outputs, conns):
    """Group nodes into evaluation layers (Kahn-style topological sort)."""
    required = required_nodes(inputs, outputs, conns)

    layers = []
    seen = set(inputs)
    while True:
        # Candidates: nodes not yet placed, all of whose inputs are available.
        candidates = {b for a, b in conns if a in seen and b not in seen}
        layer = {
            n
            for n in candidates
            if n in required and all(a in seen for a, b in conns if b == n)
        }
        if not layer:
            break
        layers.append(layer)
        seen |= layer
    return layers


class FeedForwardNetwork:
    __slots__ = ("input_keys", "output_keys", "node_evals", "_values")

    def __init__(self, input_keys, output_keys, node_evals):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.node_evals = node_evals
        self._values = {}

    @staticmethod
    def create(genome, cfg) -> "FeedForwardNetwork":
        inputs = [-i - 1 for i in range(cfg.num_inputs)]
        outputs = list(range(cfg.num_outputs))
        conns = [k for k, c in genome.conns.items() if c.enabled]

        node_evals = []
        for layer in feed_forward_layers(inputs, outputs, conns):
            for node in layer:
                links = [
                    (src, genome.conns[(src, dst)].weight)
                    for src, dst in conns
                    if dst == node
                ]
                bias = genome.nodes[node].bias if node in genome.nodes else 0.0
                node_evals.append((node, bias, links))

        # An output with no incoming connection still has a bias to apply.
        covered = {node for node, _, _ in node_evals}
        for node in outputs:
            if node not in covered:
                bias = genome.nodes[node].bias if node in genome.nodes else 0.0
                node_evals.append((node, bias, []))

        return FeedForwardNetwork(inputs, outputs, node_evals)

    def activate(self, inputs):
        values = self._values
        values.clear()
        for k, v in zip(self.input_keys, inputs):
            values[k] = v

        for node, bias, links in self.node_evals:
            total = bias
            for src, weight in links:
                total += values.get(src, 0.0) * weight
            values[node] = math.tanh(total)

        return [values.get(k, 0.0) for k in self.output_keys]
