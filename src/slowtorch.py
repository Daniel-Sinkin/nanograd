"""
A machine learning library for educational purposes. 
"""

from typing import Iterator

import numpy as np

from .nanotensor import NanoTensor
from .slowtorch_constants import RNG_SEED


class Neuron:
    def __init__(self, n_args: int = 2, seed=RNG_SEED):
        _rng = np.random.default_rng(seed)
        self.w: list[NanoTensor] = [
            NanoTensor(float(x)) for x in _rng.uniform(-1.0, 1.0, n_args)
        ]
        self.b = NanoTensor(float(_rng.uniform(-1.0, 1.0)))

    def __repr__(self):
        return f"Neuron({self.w}, {self.b})"

    def __call__(self, x: NanoTensor) -> NanoTensor:
        return self.forward(x)

    def forward(self, x: NanoTensor) -> NanoTensor:
        assert len(self.w) == len(x)
        return sum((w * x for w, x in zip(self.w, x)), self.b).tanh()


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self._neurons: list[Neuron] = [Neuron(n_in) for _ in range(n_out)]

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self._neurons)

    def __call__(self, x: NanoTensor) -> NanoTensor | list[NanoTensor]:
        retval: list[NanoTensor] = [n(x) for n in self]
        return retval[0] if len(retval) == 1 else retval


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self._layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: NanoTensor) -> NanoTensor | list[NanoTensor]:
        return self.forward(x)

    def forward(self, x: NanoTensor) -> NanoTensor | list[NanoTensor]:
        for layer in self._layers:
            x: NanoTensor | list[NanoTensor] = layer(x)
        return x
