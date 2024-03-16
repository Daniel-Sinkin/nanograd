"""Main file for the NanoGrad package."""

from src.nanotensor import NanoTensor
from src.nanotensor_autolabel import auto_label


def main():
    """Define some simple computation graph and visualize it."""
    a = NanoTensor(2.0, label="a")
    b = NanoTensor(3.0, label="b")
    c = NanoTensor(4.0, label="c")

    d = a + b
    e = b * c
    f = c.sin()
    g = f + a
    h = d + e
    i = h * g
    j = g * i
    j.backward()

    for tensor in [d, e, f, g, h, i, j]:
        auto_label(tensor)

    j.visualize_graph()


if __name__ == "__main__":
    main()
