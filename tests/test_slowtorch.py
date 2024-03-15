import numpy as np

from src.nanotensor import NanoTensor
from src.slowtorch import Neuron


def test_slowtorch_neuron() -> None:
    neuron = Neuron(2, seed=0x2024_03_15)
    x: list[NanoTensor] = [NanoTensor(-1.5), NanoTensor(2.0)]
    y: NanoTensor = neuron(x)
    z: NanoTensor = neuron.forward(x)

    assert np.isclose(y.value, 0.89789)
    assert np.isclose(y.value, z.value)
