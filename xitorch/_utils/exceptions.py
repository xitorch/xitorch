__all__ = ["GetSetParamsError", "ConvergenceWarning"]

class UnimplementedError(Exception):
    pass

class GetSetParamsError(Exception):
    pass

class ConvergenceWarning(Warning):
    """
    Warning to be raised if the convergence of an algorithm is not achieved
    """
    pass
