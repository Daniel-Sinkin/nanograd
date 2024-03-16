import random

import micrograd.engine
import micrograd.nn
import numpy as np

from src.nanotensor import NanoTensor
from src.slowtorch import MLP, Layer, Neuron


def test_slowtorch_neuron() -> None:
    """General check if the neuron init works."""
    random.seed(0x2023_03_16)

    neuron = Neuron(2)
    x: list[NanoTensor] = [NanoTensor(-1.5), NanoTensor(2.0)]
    y: NanoTensor = neuron(x)
    z: NanoTensor = neuron.forward(x)

    assert np.isclose(y.value, 0.640698)
    assert np.isclose(y.value, z.value)


def test_slowtorch_layer() -> None:
    """General check if the layer init works."""
    layer = Layer(3, 7)
    for n in layer:
        assert isinstance(n, Neuron)

    assert len(layer._neurons) == 7

    for n in layer:
        assert len(n.w) == 3


def test_slowtorch_mlp_init_against_micrograd() -> None:
    """
    Checks if our MLP gets initialized correctly.

    This test was created to be able to debug the functionality of the MLP
    but this kind of exact coupling only reall works because the nanograd tensor
    is so closely related to the micrograd implementation, if things would
    get initialized in a different order then this test would fail.

    What would be significantly better is to be able to pass an initialization
    setting to the MLPs, i.e. initial weights, and then cross-validate them.
    """

    # Might be better to be able to pass the seed directly into the init
    random.seed(0x2023_03_16)
    model_mu = micrograd.nn.MLP(2, [16, 16, 1])
    random.seed(0x2023_03_16)
    model = MLP(2, [16, 16, 1])

    assert model == model_mu

    random.seed(0x2023_03_16)
    model = MLP(20, [37, 31, 5, 12, 13])
    random.seed(0x2023_03_16)
    model_mu = micrograd.nn.MLP(20, [37, 31, 5, 12, 13])

    assert model == model_mu
