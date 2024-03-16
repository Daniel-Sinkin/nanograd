"""
Test tests used to debug the nanograd functionality by comparing them
to the micrograd functionality.
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import micrograd.engine
import micrograd.nn
import numpy as np
import torch
from micrograd.engine import Value
from sklearn.datasets import make_moons
from torch import Tensor

from src.nanotensor import NanoTensor
from src.slowtorch import MLP, Layer, Neuron


def validate_loss(model, model_mu, X, y):
    inputs_mu: list[list[Value]] = [list(map(Value, xrow)) for xrow in X]
    inputs: list[list[NanoTensor]] = [list(map(NanoTensor, xrow)) for xrow in X]

    assert all(isinstance(x, Value) for x_lst in inputs_mu for x in x_lst)
    assert all(isinstance(x, NanoTensor) for x_lst in inputs for x in x_lst)

    assert inputs_mu == inputs

    scores_mu: list[Value] = list(map(model_mu, inputs_mu))
    scores: list[NanoTensor] = list(map(model, inputs))
    assert scores == scores_mu

    losses_mu: list[Value] = [
        (1 + -yi * scorei).relu() for yi, scorei in zip(y, scores_mu)
    ]
    losses: list[NanoTensor] = [
        (1 + -yi * scorei).relu() for yi, scorei in zip(y, scores)
    ]
    assert losses == losses_mu

    data_loss_mu: Value = sum(losses_mu) * (1.0 / len(losses_mu))
    data_loss: NanoTensor = sum(losses) * (1.0 / len(losses))
    assert data_loss == data_loss_mu

    alpha = 1e-4
    reg_loss_mu: Value = alpha * sum((p * p for p in model_mu.parameters()))
    reg_loss: NanoTensor = alpha * sum((p * p for p in model.parameters))
    assert reg_loss == reg_loss_mu

    total_loss_mu: Value = data_loss_mu + reg_loss_mu
    total_loss: NanoTensor = data_loss + reg_loss
    assert total_loss == total_loss_mu

    # Note that because isinstance(np.int64, int) is not true it also follows that
    # isinstance(np.bool_, int) is not true. This is because while floats in python
    # are doubles, so that isinstance(np.float64, float) holds the ints in python
    # are actually unbounded "precision" integers.
    accuracy_mu: list[bool] = [
        bool((yi > 0) == (scorei.data > 0)) for yi, scorei in zip(y, scores_mu)
    ]
    accuracy: list[bool] = [
        bool((yi > 0) == (scorei.value > 0)) for yi, scorei in zip(y, scores)
    ]
    assert accuracy == accuracy_mu
    assert sum(accuracy) / len(accuracy) == sum(accuracy_mu) / len(accuracy_mu)
    return total_loss, accuracy, total_loss_mu, accuracy_mu


def main():
    """
    Goes through every step of the micrograd demo and validates that our
    results are the same as the ones from micrograd.

    This was created in the scope of trying to find a bug in the backpropagation,
    but during adjustments I've eventually fixed it, but I'm not sure what was,
    keeping this test here for future reference.
    """
    random.seed(0x2023_03_16)
    model_mu = micrograd.nn.MLP(2, [16, 16, 1])
    random.seed(0x2023_03_16)
    model = MLP(2, [16, 16, 1])

    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1

    model.zero_grad()
    model_mu.zero_grad()

    total_loss, acc, total_loss_mu, acc_mu = validate_loss(model, model_mu, X, y)
    total_loss.backward()
    total_loss_mu.backward()

    for layer_idx, (layer, layer_mu) in enumerate(zip(model._layers, model_mu.layers)):
        for neuron_idx, (neuron, neuron_mu) in enumerate(
            zip(layer._neurons, layer_mu.neurons)
        ):
            assert np.isclose(neuron.b.grad, neuron_mu.b.grad)
            assert np.isclose(neuron.w[0].grad, neuron_mu.w[0].grad)
            assert np.isclose(neuron.w[1].grad, neuron_mu.w[1].grad)

    learning_rate = 1.0
    for p in model.parameters:
        p.value -= learning_rate * p.grad
    for p in model_mu.parameters():
        p.data -= learning_rate * p.grad

    for layer_idx, (layer, layer_mu) in enumerate(zip(model._layers, model_mu.layers)):
        for neuron_idx, (neuron, neuron_mu) in enumerate(
            zip(layer._neurons, layer_mu.neurons)
        ):
            assert np.isclose(neuron.b.value, neuron_mu.b.data)
            assert np.isclose(neuron.w[0].value, neuron_mu.w[0].data)
            assert np.isclose(neuron.w[1].value, neuron_mu.w[1].data)

    learning_rate = 1.0 - 0.9 * 1 / 100

    model.zero_grad()
    model_mu.zero_grad()

    total_loss, acc, total_loss_mu, acc_mu = validate_loss(model, model_mu, X, y)
    total_loss.backward()
    total_loss_mu.backward()

    assert model == model_mu

    learning_rate = 1.0
    for p in model.parameters:
        p.value -= learning_rate * p.grad
    for p in model_mu.parameters():
        p.data -= learning_rate * p.grad

    assert model == model_mu


if __name__ == "__main__":
    main()
