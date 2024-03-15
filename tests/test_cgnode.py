"""Contains all the tests for the ComputationalGraph."""

from src.cgnode import (
    BinaryOperator,
    CGNodeBinary,
    CGNodeLeaf,
    CGNodeUnary,
    UnaryOperator,
)


def test_cgnode_unary() -> None:
    _n1 = CGNodeLeaf(0.7)
    _n2 = CGNodeUnary(UnaryOperator.INV, _n1)

    assert _n1.eval_() == 0.7
    assert _n2.eval_() == 1 / 0.7


def test_cgnode_binary() -> None:
    _n1 = CGNodeLeaf(0.7)
    _n2 = CGNodeLeaf(1 / 0.7)
    _n3 = CGNodeBinary(BinaryOperator.ADD, _n1, _n2)
    _n4 = CGNodeBinary(BinaryOperator.MUL, _n1, _n2)

    assert _n1.eval_() == 0.7
    assert _n2.eval_() == 1 / 0.7
    assert _n3.eval_() == 0.7 + 1 / 0.7
    assert _n4.eval_() == 0.7 * (1 / 0.7)


def test_cgnode_chained() -> None:
    _n1 = CGNodeLeaf(0.7)
    _n2 = CGNodeUnary(UnaryOperator.INV, _n1)
    _n3 = CGNodeBinary(BinaryOperator.ADD, _n1, _n2)
    _n4 = CGNodeBinary(BinaryOperator.MUL, _n3, _n2)

    assert _n4.eval_() == (0.7 + 1 / 0.7) * (1 / 0.7)
