"""
Implements the computational graph node classes.

This is deprecated and not used in the final implementation of the `nanograd` package.

The original idea was to build the actual computation graph and do the backpropagation
on an abstract graph, like how you'd do if you build out a parser for a programming
language.

I got the recursive evaluation logic working, but then decided to go with a less
abstract approach for the backpropagation.
"""

from abc import ABC, abstractmethod
from enum import Enum


class UnaryOperator(Enum):
    NEG = "-"
    INV = "inv"


class BinaryOperator(Enum):
    ADD = "+"
    MUL = "*"


class CGNode(ABC):
    """Base class for the Computation Graph"""

    @abstractmethod
    def eval_(self) -> float: ...

    @abstractmethod
    def backward(self): ...


class CGNodeLeaf(CGNode):
    """A computational graph node that is a leaf node."""

    def __init__(self, value: float):
        self.value: float = value
        self.grad = None

    def __repr__(self):
        return f"GCNode({self.value})"

    def eval_(self) -> float:
        return self.value

    def backward(self) -> float:
        if self.grad is None:
            self.grad = 1.0


class CGNodeUnary(CGNode):
    def __init__(self, operator: UnaryOperator, child: CGNode):
        self.operator: UnaryOperator = operator
        self.child: CGNode = child

    def __repr__(self):
        return f"GCNode({self.operator.value}, {self.child})"

    def eval_(self):
        match self.operator:
            case UnaryOperator.NEG:
                return -self.child.eval_()
            case UnaryOperator.INV:
                return 1 / self.child.eval_()

    def backward(self) -> float:
        match self.operator:
            case UnaryOperator.NEG:
                if self.child.grad is not None:
                    self.child.grad = -self.child.grad
                self.child.backward()
            case UnaryOperator.INV:
                return -(self.child.grad()) / (self.child.eval_() ** 2)


class CGNodeBinary(CGNode):
    """A computational graph node that is a leaf."""

    def __init__(
        self, operator: BinaryOperator, child_left: CGNode, child_right: CGNode
    ):
        self.operator: BinaryOperator = operator
        self.child_left: CGNode = child_left
        self.child_right: CGNode = child_right

    def __repr__(self):
        return f"GCNode({self.operator.value}, ({self.child_left}, {self.child_right}))"

    def eval_(self) -> float:
        match self.operator:
            case BinaryOperator.ADD:
                return self.child_left.eval_() + self.child_right.eval_()
            case BinaryOperator.MUL:
                return self.child_left.eval_() * self.child_right.eval_()

    def backward(self) -> float:
        match self.operator:
            case BinaryOperator.ADD:
                return self.child_left.grad() + self.child_right.grad()
            case BinaryOperator.MUL:
                return (
                    self.child_left.eval_() * self.child_right.grad()
                    + self.child_left.grad() * self.child_right.eval_()
                )
