import inspect
import re
import dis

def assertfcn(fcn):
    """
    Perform check on the function passed on non-linear functions.
    The function can pass the test if and only if it is:
    * a method from torch.nn.Module (that the output depends on all of the .parameters?)
    * a method from lintorch.EditableModule with the correct list of tensors
        outputted from .getparams()
    * a callable

    The function must only use variables that can be accessed from the passed inputs.
    """
    # # check if all variables are in the local scope
    # check_all_vars_in_scope(fcn)

    ismethod = inspect.ismethod(fcn)

    if not hasattr(fcn, "__call__"):
        raise RuntimeError("The fcn argument must be callable")

    # check if it is a torch.nn.module
    if ismethod and isinstance(fcn.__self__, torch.nn.Module):
        return

    # check if it is a EditableModule
    if ismethod and isinstance(fcn.__self__, EditableModule):
        return

    raise RuntimeError("The fcn must be either a function or a method from torch.nn.Module or lintorch.EditableModule")

def check_all_vars_in_scope(fcn):
    """
    Check if all variables in the fcn can be accessed from the passed inputs.
    """
    # detect if there's LOAD_GLOBAL for global variable or LOAD_DEREF for variables
    # from the outer scope
    # TODO: LOAD_DEREF and LOAD_GLOBAL turn out also called if there is a method from
    # globally imported library is called (e.g. torch.zeros)
    # This function has to check sort of thing

    s = dis.dis(fcn)
    nonloc_pattern = "LOAD_DEREF"
    glob_pattern = "LOAD_GLOBAL"
    fast_pattern = "(%s|%s)" % (nonloc_pattern, glob_pattern)
    found = re.search(fast_pattern, s)
    if not found:
        return

    # if found, raise an error with detailed message
    lineno = None
    nonloc_linenos = []
    glob_linenos = []
    nonloc_vars = []
    for line in s.split("\n"):
        elmts = line.split()

        # if the first two elements are numeric, then take the first element as the line number
        if elmts[0].isnumeric() and elmts[1].isnumeric():
            lineno = int(elmts[0])

        if nonloc_pattern in elmts:
            v = elmts[-1][1:-1]
            nonloc_linenos.append(lineno)
            nonloc_vars.append(v)

        if glob_pattern in elmts:
            v = elmts[-1][1:-1]
            glob_linenos.append(lineno)
            glob_vars.append(v)

    # write down the message (???)
