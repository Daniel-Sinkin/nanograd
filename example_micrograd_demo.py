"""
This is a modified variant of the micrograd demo from the original micrograd repository.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from src.nanotensor import NanoTensor
from src.slowtorch import MLP, Layer, Neuron


@dataclass
class SETTINGS:
    @dataclass
    class PLOT:
        MARKER_SIZE = 20
        CMAP = "jet"
        FIGSIZE: tuple[int, int] = (5, 5)

    L2ALPHA = 1e-4

    SEED = 1337


np.random.seed(SETTINGS.SEED)


def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Pulls the dataset with the same settings as in micrograd."""
    return make_moons(n_samples=100, noise=0.1)


def get_dataset_processed() -> tuple[list[NanoTensor], list[NanoTensor]]:
    """Maps 0 -> -1, 1 -> 1."""
    X, y = get_dataset()
    y = y * 2 - 1
    X = [[NanoTensor(_x[0]), NanoTensor(_x[1])] for _x in X]
    y = list(map(NanoTensor, y))
    return X, y


def plot_dataset() -> None:
    X, y = get_dataset()
    y = y * 2 - 1

    plt.figure(figsize=SETTINGS.PLOT.FIGSIZE)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=SETTINGS.PLOT.MARKER_SIZE,
        cmap=SETTINGS.PLOT.CMAP,
    )
    plt.show()


def loss(
    model: MLP, X: list[list[NanoTensor]], y: list[NanoTensor], batch_size=None
) -> tuple[float, float]:
    if batch_size is None:
        Xb, yb = X, y
    else:
        raise NotImplementedError(
            "Need to clean up the type handling before batching implementation."
        )
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]

    # forward the model to get scores
    scores = list(map(model, Xb))

    losses: list[NanoTensor] = [
        (1 - yi * scorei).relu() for yi, scorei in zip(yb, scores)
    ]
    data_loss: NanoTensor = sum(losses) * (1.0 / len(losses))

    # L2 regularization
    reg_loss: NanoTensor = SETTINGS.L2ALPHA * sum((p**2 for p in model.parameters))
    total_loss: NanoTensor = data_loss + reg_loss

    accuracy: list[bool] = [
        (yi.value > 0) == (scorei.value > 0) for yi, scorei in zip(yb, scores)
    ]
    return total_loss, sum(accuracy) / len(accuracy)


def training_iteration(model: MLP, X, y, learning_rate: float, batch_size=None) -> None:
    # Foward
    total_loss, acc = loss(model, X, y, batch_size)
    print(total_loss.value, acc)

    # Backward
    model.zero_grad()
    total_loss.backward()

    # Update via SGD
    for p in model.parameters:
        p.value -= learning_rate * p.grad


def train(model, X, y, epochs: int, batch_size=None):
    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}/{epochs}")
        learning_rate: float = 1.0 - 0.9 * epoch / 100
        training_iteration(model, X, y, learning_rate, batch_size)


def visualize_training_results(model) -> None:
    # visualize decision boundary
    h = 0.25

    X, y = get_dataset()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(NanoTensor, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.value > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == "__main__":
    # plot_dataset()
    _model = MLP(2, [16, 16, 1])
    print(len(_model.parameters))
    print(_model._layers[0]._neurons[0].w)

    train(_model, *get_dataset_processed(), epochs=100)
    visualize_training_results(_model)
