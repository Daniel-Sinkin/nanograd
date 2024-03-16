import matplotlib.pyplot as plt

from src.nanotensor import NanoTensor
from src.slowtorch import MLP, Layer, Neuron


def main():
    n = MLP(3, [4, 4, 1])
    xs: list[list[float]] = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys: list[float] = [1.0, -1.0, -1.0, 1.0]

    weights_and_biases_flattened = []
    for layer in n._layers:
        for neuron in layer._neurons:
            for wi in neuron.w:
                weights_and_biases_flattened.append(wi)
            weights_and_biases_flattened.append(neuron.b)

    def loss(y: list[NanoTensor], y_hat: list[NanoTensor]) -> NanoTensor:
        return sum(((a - b) ** 2 for a, b in zip(y, y_hat)))

    lr = 0.01

    def run_iteration():
        map(lambda x: x.zero_grad(), weights_and_biases_flattened)

        ypred = list(map(n, xs))
        _loss = loss(ypred, ys)
        _loss.backward()

        # This is horrible, can't iterate through the flattened list because of variable
        # shadowing. The correct solution would be to offload to get_parameter functions
        # but this works, so I'll leave it like this for now.
        for layer_idx in range(len(n._layers)):
            for neuron_idx in range(len(n._layers[layer_idx]._neurons)):
                for wi_idx in range(len(n._layers[layer_idx]._neurons[neuron_idx].w)):
                    n._layers[layer_idx]._neurons[neuron_idx].w[wi_idx] -= (
                        lr * n._layers[layer_idx]._neurons[neuron_idx].w[wi_idx].grad
                    )
                n._layers[layer_idx]._neurons[neuron_idx].b -= (
                    lr * n._layers[layer_idx]._neurons[neuron_idx].b.grad
                )

        return [_y.value for _y in ypred], _loss.value

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
    plt.show()


if __name__ == "__main__":
    main()
