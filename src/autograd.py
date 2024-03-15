from enum import Enum, auto
from typing import Callable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


class Operator(Enum):
    NOT_INITIALIZED = ""
    ADD = "+"
    MUL = "*"
    SIN = "sin"
    COS = "cos"
    EXP = "exp"
    POW = "pow"


class NanoTensor:
    """A simple tensor type class that supports autograd. Functions as a ndarray wrapper."""

    COUNTER = 0  # Gives every NanoTensor a unique id, does not get decremented if objects get deleted

    def __init__(
        self, value, children: tuple["NanoTensor"] = None, operator: Operator = None
    ):
        self.value = value
        self.grad: float = 0.0
        self._children: tuple["NanoTensor"] = children or ()
        self._operator: Operator = operator or Operator.NOT_INITIALIZED
        self._backward: Optional[Callable] = lambda: None
        self.label = str(NanoTensor.COUNTER)
        NanoTensor.COUNTER += 1

    def __repr__(self):
        return f"NanoTensor({self.value})"

    def __add__(self, other) -> "NanoTensor":
        if not isinstance(other, NanoTensor):
            other = NanoTensor(other)
        result_tensor = NanoTensor(
            self.value + other.value, children=(self, other), operator=Operator.ADD
        )

        def _backward() -> None:
            self.grad += 1.0 * result_tensor.grad
            other.grad += 1.0 * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def __mul__(self, other) -> "NanoTensor":
        if not isinstance(other, NanoTensor):
            other = NanoTensor(other)
        result_tensor = NanoTensor(
            self.value * other.value, children=(self, other), operator=Operator.MUL
        )

        def _backward() -> None:
            self.grad += other.value * result_tensor.grad
            other.grad += self.value * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def sin(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            np.sin(self.value), children=(self,), operator=Operator.SIN
        )

        def _backward() -> None:
            self.grad += np.cos(self.value) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def cos(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            np.cos(self.value), children=(self,), operator=Operator.COS
        )

        def _backward() -> None:
            self.grad -= np.sin(self.value) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def exp(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            np.exp(self.value), children=(self,), operator=Operator.EXP
        )

        def _backward() -> None:
            self.grad += np.exp(self.value) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def __pow__(self, power) -> "NanoTensor":
        if not isinstance(power, NanoTensor):
            power = NanoTensor(power)
        result_tensor = NanoTensor(
            self.value**power.value, children=(self, power), operator=Operator.POW
        )

        def _backward() -> None:
            self.grad += (
                power.value * self.value ** (power.value - 1)
            ) * result_tensor.grad
            power.grad += (
                self.value**power.value * np.log(self.value)
            ) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def backward(self):
        # Create a directed graph
        graph = nx.DiGraph()

        # Add nodes to the graph
        for node in self.get_all_nodes():
            graph.add_node(node)

        # Add edges to the graph
        for node in self.get_all_nodes():
            for child in node._children:
                graph.add_edge(child, node)

        # Perform topological sort
        topo_order = list(nx.topological_sort(graph))

        # dL/dL = 1.0
        self.grad = 1.0

        # Iterate through the nodes in topological order and compute gradients
        for node in reversed(topo_order):
            node._backward()

    def get_all_nodes(self) -> set["NanoTensor"]:
        """
        Returns a set of all nodes in the computational graph,
        including the current node and its children.
        """
        nodes = set()

        def traverse(node):
            nodes.add(node)
            for child in node._children:
                traverse(child)

        traverse(self)
        return nodes

    """
    def backward(self):
        topo: list["NanoTensor"] = []
        visited: set["NanoTensor"] = set()

        def build_topo(node: NanoTensor):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
    """

    def zero_grad(self) -> None:
        """Zeroes the gradient of the NanoTensor"""
        self.grad = 0.0

    def to_torch(self) -> torch.Tensor:
        """Creates a torch tensor from the NanoTensor. Does not retain gradient."""
        return torch.tensor(self.value, requires_grad=True)

    @staticmethod
    def from_torch(tensor: torch.Tensor) -> "NanoTensor":
        """Creates a NanoTensor from a torch tensor."""
        return NanoTensor(tensor.item())

    def visualize_graph(self):
        # Create a directed graph
        graph = nx.DiGraph()

        # Add nodes to the graph
        for node in self.get_all_nodes():
            graph.add_node(node, label=str(node))

        # Add edges to the graph
        for node in self.get_all_nodes():
            for child in node._children:
                graph.add_edge(child, node)

        # Draw the graph
        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color="skyblue",
            edge_color="gray",
            font_size=8,
            node_size=1000,
        )
        plt.axis("off")
        plt.show()
