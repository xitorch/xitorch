__all__ = ["GetSetParamsError", "ConvergenceWarning", "MathWarning"]

class UnimplementedError(Exception):
    pass

class GetSetParamsError(Exception):
    pass

class ConvergenceWarning(Warning):
    """
    Warning to be raised if the convergence of an algorithm is not achieved
    """
    pass

class MathWarning(Warning):
    """
    Raised if there are mathematical conditions that are not satisfied
    """
    pass
