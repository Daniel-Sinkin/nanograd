"""Contains all the tests for the NanoTensor and in particular autograd functionality."""

import numpy as np
import torch

from src.nanotensor import NanoTensor


def test_autograd_add() -> None:
    x = NanoTensor(3.0)
    y = NanoTensor(4.0)
    z: NanoTensor = x + y
    z.backward()

    assert z.value == 7.0
    assert x.grad == 1.0
    assert y.grad == 1.0


def test_autograd_mul() -> None:
    x = NanoTensor(3.0)
    y = NanoTensor(4.0)
    z: NanoTensor = x * y
    z.backward()

    x_t: torch.Tensor = x.to_torch()
    y_t: torch.Tensor = y.to_torch()
    z_t: torch.Tensor = x_t * y_t
    z_t.backward()

    assert z.value == z_t.item()
    assert x.grad == x_t.grad.item()
    assert y.grad == y_t.grad.item()


def test_autograd_sin() -> None:
    x = NanoTensor(0.5)
    y: NanoTensor = x.sin()
    y.backward()

    x_t: torch.Tensor = x.to_torch()
    y_t: torch.Tensor = x_t.sin()
    y_t.backward()

    assert np.isclose(y.value, y_t.item())
    assert np.isclose(x.grad, x_t.grad.item())


def test_autograd_cos() -> None:
    x = NanoTensor(0.5)
    y: NanoTensor = x.cos()
    y.backward()

    x_t: torch.Tensor = x.to_torch()
    y_t: torch.Tensor = x_t.cos()
    y_t.backward()

    assert np.isclose(y.value, y_t.item())
    assert np.isclose(x.grad, x_t.grad.item())


def test_autograd_validate_chained() -> None:
    x = NanoTensor(3.0)
    y = NanoTensor(4.0)
    z = NanoTensor(5.0)

    a: NanoTensor = x * y
    b: NanoTensor = a + z
    b.backward()

    x_t: torch.Tensor = x.to_torch()
    y_t: torch.Tensor = y.to_torch()
    z_t: torch.Tensor = z.to_torch()
    a_t: torch.Tensor = x_t * y_t
    b_t: torch.Tensor = a_t + z_t
    b_t.backward()

    assert b.value == b_t.item()
    assert x.grad == x_t.grad.item()
    assert y.grad == y_t.grad.item()
    assert a.value == a_t.item()


def test_autograd_exp() -> None:
    x = NanoTensor(0.5)
    y: NanoTensor = x.exp()
    y.backward()

    x_t: torch.Tensor = x.to_torch()
    y_t: torch.Tensor = x_t.exp()
    y_t.backward()

    assert np.isclose(y.value, y_t.item())
    assert np.isclose(x.grad, x_t.grad.item())


def test_autograd_pow() -> None:
    x = NanoTensor(2.0)
    y = 3.0
    z: NanoTensor = x**y
    z.backward()

    x_t: torch.Tensor = x.to_torch()
    z_t: torch.Tensor = x_t**3.0
    z_t.backward()

    assert np.isclose(z.value, z_t.item())
    assert np.isclose(x.grad, x_t.grad.item())

    x.zero_grad()
    x_t.grad.zero_()


def test_autograd_pow_int_normal() -> None:
    x = NanoTensor(2.0)
    y = 3
    z: NanoTensor = x**y
    z.backward()

    x_t: torch.Tensor = x.to_torch()
    z_t: torch.Tensor = x_t**y
    z_t.backward()

    assert np.isclose(z.value, z_t.item())
    assert np.isclose(x.grad, x_t.grad.item())


def test_autograd_pow_int_zero_exponent() -> None:
    x = NanoTensor(1.0)
    y = 0
    z: NanoTensor = x**y
    z.backward()

    x_t: torch.Tensor = x.to_torch()
    z_t: torch.Tensor = x_t**y
    z_t.backward()

    assert np.isclose(z.value, z_t.item())
    assert np.isclose(x.grad, x_t.grad.item())


def test_autograd_tanh() -> None:
    x = NanoTensor(0.5)
    y: NanoTensor = x.tanh()
    y.backward()

    x_t: torch.Tensor = x.to_torch()
    y_t: torch.Tensor = x_t.tanh()
    y_t.backward()

    assert np.isclose(y.value, y_t.item())
    assert np.isclose(x.grad, x_t.grad.item())


def test_autograd_relu_positive() -> None:
    x_pos = NanoTensor(1.0)
    y_pos: NanoTensor = x_pos.relu()
    y_pos.backward()

    x_pos_t: torch.Tensor = x_pos.to_torch()
    y_pos_t: torch.Tensor = torch.relu(x_pos_t)
    y_pos_t.backward()

    assert np.isclose(
        y_pos.value, y_pos_t.item()
    ), "ReLU forward pass failed for positive value."
    assert np.isclose(
        x_pos.grad, x_pos_t.grad.item()
    ), "ReLU backward pass failed for positive value."


def test_autograd_relu_negative() -> None:
    x_neg = NanoTensor(-1.0)
    y_neg: NanoTensor = x_neg.relu()
    y_neg.backward()

    x_neg_t: torch.Tensor = x_neg.to_torch()
    y_neg_t: torch.Tensor = torch.relu(x_neg_t)
    y_neg_t.backward()

    assert np.isclose(
        y_neg.value, y_neg_t.item()
    ), "ReLU forward pass failed for negative value."
    assert np.isclose(
        x_neg.grad, x_neg_t.grad.item()
    ), "ReLU backward pass failed for negative value."


def test_autograd_edge_cases() -> None:
    x = NanoTensor(3.0)
    y: NanoTensor = x + x
    y.backward()
    assert x.grad == 2.0
