import os
import random

import matplotlib.pyplot as plt

from src.nanotensor import NanoTensor
from src.slowtorch import MLP, Layer, Neuron


def main(show_plot=True, filepath: str = None):
    random.seed(0x2024_03_16)

    model = MLP(3, [4, 4, 1])
    xs: list[list[float]] = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys: list[float] = [1.0, -1.0, -1.0, 1.0]

    def loss(y: list[NanoTensor], y_hat: list[NanoTensor]) -> NanoTensor:
        return sum(((a - b) ** 2 for a, b in zip(y, y_hat)))

    lr = 0.01

    def run_iteration() -> tuple[list[float], float]:
        model.zero_grad()

        ypred = list(map(model, xs))
        _loss = loss(ypred, ys)
        _loss.backward()

        for p in model.parameters:
            p.value -= lr * p.grad

        return [float(_y.value) for _y in ypred], _loss.value

    lst = [run_iteration() for _ in range(100)]

    N = len(lst)
    x = range(N)
    # Create a figure and add subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot on the first subplot
    ax1.plot(x, [y[0] for y, _ in lst], c="r", label="y[0]")
    ax1.plot(x, [y[1] for y, _ in lst], c="b", label="y[1]")
    ax1.plot(x, [y[2] for y, _ in lst], c="black", label="y[2]")
    ax1.plot(x, [y[3] for y, _ in lst], c="g", label="y[3]")
    ax1.set_title("y_pred values")
    ax1.legend()

    # Plot the loss (second element) on the second subplot
    ax2.plot(x, [_loss for _, _loss in lst], c="b", label="Loss")
    ax2.set_title("Loss")
    ax2.legend()

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()


if __name__ == "__main__":
    main(
        show_plot=True,
        filepath=os.path.join("images", "slowtorch_binary_classification.png"),
    )
