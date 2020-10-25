import torch
from xitorch._core.editable_module import EditableModule
from xitorch._impls.integrate.samples_quad import CubicSplineSQuad, TrapzSQuad, SimpsonSQuad
from xitorch._utils.misc import get_method
from xitorch._docstr.api_docstr import get_methods_docstr
from typing import List, Union, Callable

__all__ = ["SQuad"]

class SQuad(EditableModule):
    r"""
    SQuad (Sampled QUADrature) is a class for quadrature performed with a
    fixed samples at given points.
    Mathematically, it does the integration

    .. math::

        \mathbf{z}(x) = \int_{x_0}^x \mathbf{y}(x')\ \mathrm{d}x

    where :math:`\mathbf{y}(x)` is the interpolated function from a given sample.

    Arguments
    ---------
    x: torch.Tensor
        The positions where the samples are given. It is a 1D tensor with shape
        ``(nx,)``.
    method: str or callable or None
        The integration method. If None, it will choose ``"cspline"``.
    **fwd_options
        Method-specific options (see method section below)
    """

    def __init__(self, x: torch.Tensor,
                 method: Union[str, Callable, None] = None,
                 **fwd_options):
        if method is None:
            method = "cspline"
        if not (isinstance(x, torch.Tensor) and len(x.shape) == 1):
            raise RuntimeError("The input x to SQuad must be a 1D tensor")

        all_clss = {
            "cspline": CubicSplineSQuad,
            "simpson": SimpsonSQuad,
            "trapz": TrapzSQuad,
        }
        clss = get_method("SQuad", all_clss, method)
        self.obj = clss(x, **fwd_options)
        self.nx = x.shape[-1]

    def cumsum(self, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        r"""
        Perform the cumulative integration of the samples :math:`\mathbf{y}`
        over the specified dimension.

        Arguments
        ---------
        y: torch.Tensor
            The value of samples. The size of ``y`` at ``dim`` must be equal
            to the length of ``x``.
        dim: int
            The dimension where the cumulative integration is performed.

        Returns
        -------
        torch.Tensor
            The cumulative integrated values with the same shape as ``y``.
        """
        swapaxes = dim != -1
        if swapaxes:
            y = y.transpose(dim, -1)
        if y.shape[-1] != self.nx:
            raise RuntimeError("The length of integrated dimension does not match with x")
        res = self.obj.cumsum(y)
        if swapaxes:
            res = res.transpose(dim, -1)
        return res

    def integrate(self, y: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        r"""
        Perform the full integration of the samples :math:`\mathbf{y}`
        over the specified dimension.

        Arguments
        ---------
        y: torch.Tensor
            The value of samples. The size of ``y`` at ``dim`` must be equal
            to the length of ``x``, i.e. ``(..., nx, ...)``.
        dim: int
            The dimension where the integration is performed.
        keepdim: bool
            Option to not discard the integrated dimension. If ``True``, the
            integrated dimension size will be 1.

        Returns
        -------
        torch.Tensor
            The integrated values.
        """
        swapaxes = dim != -1
        if swapaxes:
            y = y.transpose(dim, -1)
        if y.shape[-1] != self.nx:
            raise RuntimeError("The length of integrated dimension does not match with x")
        res = self.obj.integrate(y)
        if keepdim:
            res = res.unsqueeze(-1)
        if swapaxes:
            res = res.transpose(dim, -1)
        return res

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """"""
        return self.obj.getparamnames(methodname, prefix=prefix + "obj.")


# docstring completion
_squad_methods = {
    "cspline": CubicSplineSQuad,
    # "simpson": SimpsonSQuad,
    "trapz": TrapzSQuad,
}
SQuad.__doc__ = get_methods_docstr(SQuad, _squad_methods)
