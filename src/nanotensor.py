from enum import Enum, auto
from typing import Callable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .nanotensor_constants import Operator
from .nanotensor_util import format_label


class NanoTensor:
    """A simple tensor type class that supports autograd. Functions as a ndarray wrapper."""

    COUNTER = 0  # Gives every NanoTensor a unique id, does not get decremented if objects get deleted

    def __init__(
        self,
        value,
        children: tuple["NanoTensor"] = None,
        operator: Operator = None,
        label: str = None,
    ):
        self.value = value
        self.grad: float = 0.0
        self._children: tuple["NanoTensor"] = children or ()
        self._operator: Operator = operator or Operator.NOT_INITIALIZED
        self._backward: Optional[Callable] = lambda: None

        self.label: str = label or str(NanoTensor.COUNTER)
        NanoTensor.COUNTER += 1

    def __repr__(self):
        return f"NanoTensor({self.value},{self.label})"

    def __add__(self, other) -> "NanoTensor":
        if not isinstance(other, NanoTensor):
            other = NanoTensor(other)
        result_tensor = NanoTensor(
            self.value + other.value, children=(self, other), operator=Operator.ADD
        )

        # d/dx +(x, y) = 1, d/dy +(x, y) = 1
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

        # d/dx *(x, y) = y, d/dy *(x, y) = x
        def _backward() -> None:
            self.grad += other.value * result_tensor.grad
            other.grad += self.value * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def sin(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            np.sin(self.value), children=(self,), operator=Operator.SIN
        )

        # d/dx sin(x) = cos(x)
        def _backward() -> None:
            self.grad += np.cos(self.value) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def cos(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            np.cos(self.value), children=(self,), operator=Operator.COS
        )

        # d/dx cos(x) = -sin(x)
        def _backward() -> None:
            self.grad -= np.sin(self.value) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def exp(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            np.exp(self.value), children=(self,), operator=Operator.EXP
        )

        # d/dx exp(x) = exp(x)
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

        # d/dx x^y = y * x^(y-1), d/dy x^y = x^y * ln(x)
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
        """Backpropagate gradients through the computational graph."""
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

    def visualize_graph(self, filepath: str = None):
        """
        Displays a visualization of the computational graph.

        Add a filepath if you want to save the figure, leave as None to only display it.

        Potential Improvements:
            - Make the nodes first point into a "synthetic" red node that only
              contains the operator which then points into the value and grad.
            - Sort the nodes within a given layer, for example so that a < b < c
              gets preserved. In particular this would make the graph be stable
              to calling backpropagation on an intermediate node, which would
              allow for a more interactive visualization.
            - Clean the graph up, making labels, adding more information, give
              some dashboard type of information on the side.
        """
        graph = nx.DiGraph()
        for node in self.get_all_nodes():
            graph.add_node(node, label=format_label(node))
        for node in self.get_all_nodes():
            for child in node._children:
                graph.add_edge(child, node)

        # Initialize layer information
        layers: dict["NanoTensor", int] = {node: 0 for node in graph.nodes()}

        # Update layers based on topological sorting
        for node in nx.topological_sort(graph):
            preds = list(graph.predecessors(node))
            if preds:  # If there are predecessors
                layers[node] = max(layers[pred] + 1 for pred in preds)

        # Organize nodes by layers to calculate positions
        layer_counts = {}
        for layer in layers.values():
            if layer not in layer_counts:
                layer_counts[layer] = 0
            layer_counts[layer] += 1

        # Assign positions based on layers
        pos = {}
        layer_positions = {
            layer: 0 for layer in layer_counts
        }  # Tracks position within each layer
        for node, layer in sorted(layers.items(), key=lambda x: x[1]):  # Sort by layer
            width = layer_counts[layer]
            pos[node] = (
                layer_positions[layer] - width / 2 + 0.5,
                -layer,
            )  # Center align within layer
            layer_positions[layer] += 1

        nx.draw(graph, pos, node_color="black", node_size=2250, edgelist=[])
        nx.draw(
            graph,
            pos,
            labels=nx.get_node_attributes(graph, "label"),
            with_labels=True,
            node_color="skyblue",
            edge_color="gray",
            font_size=8,
            node_size=2000,
        )
        plt.axis("off")
        if filepath is not None:
            plt.savefig(filepath)
        plt.show()
