from enum import Enum

UNICODE_NABLA = "\u2207"


class Operator(Enum):
    NOT_INITIALIZED = "LEAF"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    SIN = "sin"
    COS = "cos"
    EXP = "exp"
    POW = "pow"
    TANH = "tanh"
