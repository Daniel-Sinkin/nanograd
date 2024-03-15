"""Contains code for experimenting with autograd in PyTorch."""

import torch
from torch import Tensor


def computing_chain_rule_direct() -> None:
    """
    Computes the derivative of y = sin(x) * exp(x) using the chain rule directly
    and checks that those two derivatives are actually equal.

    Note that if z = f(x, y) is some computation and we do z.backward() then
    x.grad = dz/dx and y.grad = dz/dy.

    This makes sense from a memory perspective but is exactly the opposite of
    how it would make intuitive sense, where we'd save the gradient of z
    as a 2 dimensional vector w.grad == (x.grad, y.grad).
    """
    x: torch.Tensor = torch.tensor([2.0], requires_grad=True)

    y1: Tensor = torch.sin(x)
    y2: Tensor = torch.exp(x)
    y: Tensor = y1 * y2
    y.backward()
    grad_direct: Tensor = x.grad.clone()

    x: torch.Tensor = torch.tensor([2.0], requires_grad=True)

    y1 = torch.sin(x)
    y1.backward()
    grad1: Tensor = x.grad.clone()

    x.grad.zero_()

    y2 = torch.exp(x)
    y2.backward()
    grad2: Tensor = x.grad.clone()

    x.grad.zero_()

    grad_indirect: Tensor = (y1 * grad2 + grad1 * y2).clone()

    assert grad_direct == grad_indirect


def computing_chain_rule_for_product() -> None:
    x = torch.tensor([0.7], requires_grad=True)
    y = torch.tensor([0.5], requires_grad=True)
    c = torch.cos(x + y)
    s = torch.sin(x - y)
    z = c * s
    z.backward()
    print(x.grad, y.grad)
