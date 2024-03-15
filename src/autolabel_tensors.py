"""
Utility to automatically label tensors with their variable name.

This is highly inefficient and error prone and should only be used for
showcasing things, not for actual functionality.

The labels are currently simply assigned a number equal to the total number
of tensors created so far.
"""

import inspect


def auto_label(tensor):
    """
    Automatically assign a label to a NanoTensor instance based on the variable name
    in the caller's local scope. This function uses the Python inspect module to
    find a name in the caller's local variables that matches the tensor's id.
    """
    # Get the frame for the caller of this function
    caller_frame = inspect.currentframe().f_back
    # Access the local variables of the caller
    local_vars = caller_frame.f_locals.items()
    # Search for a variable name that matches the tensor's id
    for name, var in local_vars:
        if id(var) == id(tensor):
            # Assign the found name as the label of the tensor
            tensor.label = name
            break
