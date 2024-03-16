from enum import Enum, auto
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import micrograd.engine
import networkx as nx
import numpy as np
import torch

from .nanotensor_constants import Operator
from .nanotensor_util import format_label


class NanoTensor:
    """
    A simple tensor type class that supports autograd. Only supports scalar values.

    It's called "Nano"Tensor because it's a wrapper around float (or float64)
    instead of supporting general higher-dimension data layouts.

    Implementing those might actually be feasible if we wrap numpy ndarrays instead
    of doubles but then we'd also have to implement logic to reduce the jacobian
    tensor that the gradient would end up being.

    Also would have to rework how we handle the backpropagation logic itself, to
    make a reduceable jacobian as the gradient instead of just a Tensor.
    At that point it might be better to just write the whole library from scratch.
    """

    COUNTER = 0  # Gives every NanoTensor a unique id, does not get decremented if objects get deleted

    def __init__(
        self,
        value,
        children: tuple["NanoTensor"] = None,
        operator: Operator = None,
        label: str = None,
    ):
        self.value = float(value)
        self.grad: float = 0
        self._children: tuple["NanoTensor"] = children or ()

        # Only for displaying the graphics
        self._operator: Operator = operator or Operator.NOT_INITIALIZED

        # The backpropagation function, gets assigned at the forward propagation
        # and then invoked by the backtracking.
        self._backward: Optional[Callable] = lambda: None

        # Gives each tensor a unique enumeration label for displaying them.
        self.label: str = label or str(NanoTensor.COUNTER)
        NanoTensor.COUNTER += 1

    def __repr__(self):
        return f"NanoTensor({self.value},{self.grad}])"

    def __add__(self, other) -> "NanoTensor":
        """
        We assume that we only add float or Tensor objects to Tensor
        and so we are free to wrap the floats into a new Tensor.
        """
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

    def __radd__(self, other) -> "NanoTensor":
        return self.__add__(other)

    def __sub__(self, other) -> "NanoTensor":
        if not isinstance(other, NanoTensor):
            other = NanoTensor(other)
        result_tensor = NanoTensor(
            self.value - other.value, children=(self, other), operator=Operator.SUB
        )

        # d/dx -(x, y) = 1, d/dy -(x, y) = -1
        def _backward() -> None:
            self.grad += 1.0 * result_tensor.grad
            other.grad -= 1.0 * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def __rsub__(self, other) -> "NanoTensor":
        return (-1) * self.__sub__(other)

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

    def __rmul__(self, other) -> "NanoTensor":
        return self.__mul__(other)

    def __neg__(self) -> "NanoTensor":
        result_tensor = NanoTensor(-self.value, children=(self,), operator=Operator.NEG)

        # d/dx -x = -1
        def _backward() -> None:
            self.grad -= 1.0 * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def __eq__(self, other):
        if isinstance(other, NanoTensor):
            return (self.value == other.value) and (self.grad == other.grad)

        if isinstance(other, micrograd.engine.Value):
            return np.isclose(self.value, other.data) and np.isclose(
                self.grad, other.grad
            )

        return NotImplemented

    # Have to implement it explicitly because __eq__ disables hashing
    def __hash__(self):
        return hash((self.value, self.label))

    # In principle we could replace the (globally) analytical functions
    # exp, sin, cos, tanh by a finite approximation via their power series expansion
    # e^x = sum_k^N x^k / k!, sin(x) = sum_k^N ..., cos(x)^N = sum_k ...
    # then we'd only need to implement the pow primitive.

    def exp(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            float(np.exp(self.value)), children=(self,), operator=Operator.EXP
        )

        # d/dx exp(x) = exp(x), after all it's the unique solution to y = y', y(0) = 1
        def _backward() -> None:
            self.grad += float(np.exp(self.value)) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    # If we'd support complex types we could build everything from the
    # exponential function as a primitive because
    # sin(x) = (e^ix - e^-ix) / 2i
    # cos(x) = (e^ix + e^ix) / 2

    def sin(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            float(np.sin(self.value)), children=(self,), operator=Operator.SIN
        )

        # d/dx sin(x) = cos(x)
        def _backward() -> None:
            self.grad += float(np.cos(self.value)) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def cos(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            float(np.cos(self.value)), children=(self,), operator=Operator.COS
        )

        # d/dx cos(x) = -sin(x)
        def _backward() -> None:
            self.grad -= float(np.sin(self.value)) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def tanh(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            float(np.tanh(self.value)), children=(self,), operator=Operator.TANH
        )

        # d/dx tanh(x) = 1 - tanh(x)^2
        def _backward() -> None:
            self.grad += (1 - result_tensor.value**2) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    # We could implement support for general piecewise funcitons by passing in
    # a collection of function points and partitions into a higher order function,
    # for example then we'd have relu = piecewise([_zero, _id])([0])
    # and for x <= 0 then relu(x) = _zero(x) = 0
    # and for x > 0  then relu(x) = _id(x) = x
    # piecewise : List[Callable[float, float]] -> List[float] -> float
    # That with the power series argument would reduce our set of primitives
    # to only polynomials.

    def relu(self) -> "NanoTensor":
        result_tensor = NanoTensor(
            max(self.value, 0),
            children=(self,),
            operator=Operator.RELU,
        )

        # d/dx relu(x) is a piecewise function that is 1.0 on (0, inf) and
        # equal to 0 for (-inf, 0]. Note that it's not actually differentiable
        # around 0, probably only a theoretical concern.
        def _backward() -> None:
            self.grad += (1.0 if self.value > 0 else 0.0) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    def __pow__(self, power) -> "NanoTensor":
        if not isinstance(power, int | float):
            raise TypeError("Exponent must be an int or a float")
        result_tensor = NanoTensor(
            self.value**power, children=(self,), operator=Operator.POW
        )

        # d/dx x^y = y * x^(y-1)
        # For us x^y is not a binary map pow: R x R -> R but instead denotes
        # a family of functions {pow(y) : y in R} for y in R where pow(y)(x) = x^y.
        # As such we are not interested in d/dy x^y, but it would be the following:
        # d/dy x^y = d/dy e^(ln(x) y) = ln(x) e^(ln(x) y) = ln(x)x^y
        def _backward() -> None:
            self.grad += (power * self.value ** (power - 1)) * result_tensor.grad

        result_tensor._backward = _backward
        return result_tensor

    """
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    """

    def backward(self):
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

    # Could implement a general interface to allow for interoperability with
    # different Tensor representations, maybe even just list pytorch tensors
    # use NanoTensor <--> torch.Tensor <--> everything else

    # NanoTensor ->  to_torch  -> torch.Tensor -> ...
    # NanoTensor <- from_torch <- torch.Tensor <- ...

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

        # Draws outline around the nodes
        nx.draw(graph, pos, node_color="black", node_size=2250, edgelist=[])
        # Draws the actual nodes
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

    # This could be changed to a generic tensor -> nanotensor method that handles
    # nesting properly, but as we don't support non-scalar Tensor primitives
    # this wouldn't really be that helpful.
    @staticmethod
    def from_list(lst: list) -> list["NanoTensor"]:
        return [NanoTensor(x) for x in lst]

    @staticmethod
    def from_nested_list(lst: list[list]) -> list[list["NanoTensor"]]:
        return [NanoTensor.from_list(sublst) for sublst in lst]
