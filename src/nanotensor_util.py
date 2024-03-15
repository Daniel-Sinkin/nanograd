from .nanotensor_constants import UNICODE_NABLA, Operator


def format_label(node) -> str:
    # Base label part common to all nodes
    base_label = f"[{node.label}]\n{node.value:.2f}\n{UNICODE_NABLA}={node.grad:.2f}"

    # For nodes without children (leaf nodes)
    if len(node._children) == 0:
        return base_label

    # Generate operation part of the label based on the number of children and the operator type
    if len(node._children) == 1:
        # Unary operation format
        operation_part = f"{node._operator.value}([{node._children[0].label}])"
    elif len(node._children) == 2:
        # Check if the operation is infix (e.g., addition, multiplication) or not
        if node._operator in [Operator.ADD, Operator.MUL]:
            # Infix format for binary operations like addition and multiplication
            operation_part = f"[{node._children[0].label}] {node._operator.value} [{node._children[1].label}]"
        else:
            # Prefix format for other binary operations
            operation_part = f"{node._operator.value}([{node._children[0].label}], [{node._children[1].label}])"
    else:
        # Fallback for nodes with more than two children or unspecified operations
        operation_part = node._operator.value if node._operator else "Unknown Op"

    # Combine the base label with the operation part
    label = f"{base_label}\n{operation_part}"
    return label
