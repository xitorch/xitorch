import torch
from xitorch._core.editable_module import EditableModule
from xitorch._impls.interpolate.cubic_spline_1d import CubicSpline1D

__all__ = ["Interp1D"]

class Interp1D(EditableModule):
    """
    1D interpolation class. When initializing the class, the `x` must be
    specified and `y` can be specified during initialization or later.

    __init__ arguments
    ------------------
    * x: torch.Tensor (nr,)
        The position of known values
    * y: torch.Tensor (*BY, nr) or None
        The known values. If None, it must be supplied during __call__
    * method: str or None
        Interpolation method
    * **fwd_options: dict
        The method-specific options

    __call__ arguments
    ------------------
    * xq: torch.Tensor (nrq,)
        The position of query points.
    * y: torch.Tensor (*BY, nr) or None
        The values. If `y` has been specified during __init__ and here it is
        not None, this value of `y` will be ignored.
        If no `y` ever specified, then it will raise an error.
    """
    def __init__(self, x, y=None, method=None, **fwd_options):
        if method is None:
            method = "cspline"
        if method == "cspline":
            self.obj = CubicSpline1D(x, y)
        else:
            raise RuntimeError("Unknown interp1d method: %s" % method)

    def __call__(self, xq, y=None):
        return self.obj(xq, y)

    def getparamnames(self, methodname, prefix=""):
        return [prefix+c for c in self.obj.getparamnames()]
