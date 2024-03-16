"""
A machine learning library for educational purposes. 
"""

import random
from abc import ABC, abstractmethod
from typing import Iterator

import micrograd.nn
import numpy as np

from .nanotensor import NanoTensor


class Module(ABC):
    """Baseclass for Neuron, Layer and MLP."""

    @abstractmethod
    def forward(self, x: NanoTensor) -> NanoTensor | list[NanoTensor]: ...

    def __call__(self, x: NanoTensor) -> NanoTensor | list[NanoTensor]:
        return self.forward(x)

    @property
    @abstractmethod
    def parameters(self) -> list[NanoTensor]: ...

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Neuron(Module):
    def __init__(self, n_args: int = 2, nonlin=True):
        self.w: list[NanoTensor] = [
            NanoTensor(random.uniform(-1, 1)) for _ in range(n_args)
        ]
        self.b = NanoTensor(0)
        self.nonlin = nonlin

    def __repr__(self):
        return f"Neuron({self.w}, {self.b})"

    @property
    def parameters(self) -> list[NanoTensor]:
        return self.w + [self.b]

    def forward(self, x: NanoTensor) -> NanoTensor:
        assert len(self.w) == len(x)
        _forward: NanoTensor = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return _forward.relu() if self.nonlin else _forward

    def __eq__(self, other):
        if isinstance(other, micrograd.nn.Neuron):
            return (
                self.w == other.w and self.b == other.b and self.nonlin == other.nonlin
            )

        return NotImplemented


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        self._neurons: list[Neuron] = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self._neurons)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def __getitem__(self, idx: int) -> Neuron:
        return self._neurons[idx]

    def __eq__(self, other):
        # Iterators are equal if all(zip(__eq__)) is true, i.e. for lists it
        # does the n.__eq__(np) for all the elements in the corresponding lists
        if isinstance(other, micrograd.nn.Layer):
            return self._neurons == other.neurons

        return NotImplemented

    def forward(self, x: NanoTensor) -> NanoTensor | list[NanoTensor]:
        retval: list[NanoTensor] = [n(x) for n in self]
        return retval[0] if len(retval) == 1 else retval

    @property
    def parameters(self) -> list[NanoTensor]:
        return [param for neuron in self for param in neuron.parameters]


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]):
        sz: list[int] = [nin] + nouts
        self._layers: list[Layer] = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __getitem__(self, idx: int) -> Layer:
        return self._layers[idx]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._layers)

    def __eq__(self, other):
        if isinstance(other, micrograd.nn.MLP):
            return self._layers == other.layers

        return NotImplemented

    def forward(self, x: NanoTensor) -> NanoTensor | list[NanoTensor]:
        """Returns a NanoTensor if the model has a single output, else a list of NanoTensors."""
        for layer in self._layers:
            x: NanoTensor | list[NanoTensor] = layer(x)
        return x

    @property
    def parameters(self) -> list[NanoTensor]:
        return [param for layer in self for param in layer.parameters]

    def hinge_loss(
        self,
        X: list[list[NanoTensor]],
        y: list[NanoTensor],
        alpha=1e-4,
        batch_size=None,
    ) -> tuple[float, float]:
        if batch_size is None:
            Xb, yb = X, y
        else:
            raise NotImplementedError(
                "Need to clean up the type handling before batchign implementation."
            )
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]

        # forward the model to get scores
        scores = list(map(self, Xb))

        losses: list[NanoTensor] = [
            (1 - yi * scorei).relu() for yi, scorei in zip(yb, scores)
        ]
        data_loss: NanoTensor = sum(losses) * (1.0 / len(losses))

        # L2 regularization
        reg_loss: NanoTensor = alpha * sum((p**2 for p in self.parameters))
        total_loss: NanoTensor = data_loss + reg_loss

        accuracy: list[bool] = [
            (yi.value > 0) == (scorei.value > 0) for yi, scorei in zip(yb, scores)
        ]
        return total_loss, sum(accuracy) / len(accuracy)
