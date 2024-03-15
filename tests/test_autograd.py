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
    y = NanoTensor(3.0)
    z: NanoTensor = x**y
    z.backward()

    x_t: torch.Tensor = x.to_torch()
    y_t: torch.Tensor = y.to_torch()
    z_t: torch.Tensor = x_t**y_t
    z_t.backward()

    assert np.isclose(z.value, z_t.item())
    assert np.isclose(x.grad, x_t.grad.item())
    assert np.isclose(y.grad, y_t.grad.item())

    x.zero_grad()
    y.zero_grad()
    x_t.grad.zero_()
    y_t.grad.zero_()


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


def test_autograd_edge_cases() -> None:
    x = NanoTensor(3.0)
    y: NanoTensor = x + x
    y.backward()
    assert x.grad == 2.0
