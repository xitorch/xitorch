from typing import Optional, List, Union, Callable
import torch
from xitorch._core.editable_module import EditableModule
from xitorch._impls.interpolate.interp_1d import CubicSpline1D, LinearInterp1D
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch._utils.misc import get_method

__all__ = ["Interp1D"]

class Interp1D(EditableModule):
    """
    1D interpolation class. When initializing the class, the `x` must be
    specified and `y` can be specified during initialization or later.

    Arguments
    ---------
    x: torch.Tensor
        The position of known values in tensor with shape ``(nr,)``
    y: torch.Tensor or None
        The values at the given position with shape ``(*BY, nr)``.
        If ``None``, it must be supplied during ``__call__``
    method: str or callable or None
        Interpolation method. If None, it will choose ``"cspline"``.
    **fwd_options
        Method-specific options (see method section below)
    """

    def __init__(self,
                 x: torch.Tensor,
                 y: Optional[torch.Tensor] = None,
                 method: Union[str, Callable, None] = None,
                 **fwd_options):
        if method is None:
            method = "cspline"
        methods = {
            "cspline": CubicSpline1D,
            "linear": LinearInterp1D,
        }
        method_cls = get_method("Interp1D", methods, method)
        self.obj = method_cls(x, y, **fwd_options)

    def __call__(self, xq: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Arguments
        ----------------
        xq: torch.Tensor
            The position of query points with shape ``(nrq,)``.
        y: torch.Tensor or None
            The values at the given position with shape ``(*BY, nr)``.
            If ``y`` has been specified during ``__init__`` and also
            specified here, the value of ``y`` given here will be ignored.
            If no ``y`` ever specified, then it will raise an error.

        Returns
        -------
        torch.Tensor
            The interpolated values with shape ``(*BY, nrq)``.
        """
        return self.obj(xq, y)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """"""
        return [prefix + "obj." + c for c in self.obj.getparamnames()]


# docstring completion
interp1d_methods = {
    "cspline": CubicSpline1D,
    "linear": LinearInterp1D,
}
Interp1D.__doc__ = get_methods_docstr(Interp1D, interp1d_methods)
