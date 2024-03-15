"""
This is an exact copy of the plotting logic that can be found in 
https://github.com/karpathy/micrograd/
"""

from graphviz import Digraph

from src.nanotensor import NanoTensor
from src.slowtorch import MLP, Layer, Neuron


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ data %.4f | grad %.4f }" % (n.value, n.grad),
            shape="record",
        )
        if n._operator.value != "":
            dot.node(name=str(id(n)) + n._operator.value, label=n._operator.value)
            dot.edge(str(id(n)) + n._operator.value, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._operator.value)

    return dot
