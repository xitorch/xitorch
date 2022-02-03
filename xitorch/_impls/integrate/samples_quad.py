import torch
import xitorch as xt
from abc import abstractmethod
from xitorch._impls.interpolate.interp_1d import _get_spline_mat_inv

"""
This file contains the cumulative sum quadrature functions.
The functions are usually used in solve_poisson method in grid objects.
"""

class BaseSQuad(xt.EditableModule):
    @abstractmethod
    def cumsum(self, y):
        """
        Cumsum the last dimension.
        """
        pass

    @abstractmethod
    def integrate(self, y):
        """
        Integrate the last dimension without keeping the dimension.
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname, prefix=""):
        pass

class CubicSplineSQuad(BaseSQuad):
    r"""
    Perform integration of given sampled values by assuming it is interpolated
    with cubic spline [1]_. It is simply

    .. math::

        S = \sum_{i=0}^{N-2} \left[\frac{1}{2}(y_i+y_{i+1}) + \frac{1}{12}(y'_i - y'_{i+1})(x_{i+1}-x_i)^2\right]

    Keyword arguments
    -----------------
    bc_type: str
        Boundary condition. See :class:`xitorch.interpolate.Interp1D` with
        ``"cspline"`` method for details.

    References
    ----------
    .. [1] Mark H. Holmes, "Connections Between Cubic Splines and Quadrature Rules" (eq. 8),
           The American Mathematical Monthly, Volume 121, Issue 8, 2014.
    """

    def __init__(self, x, bc_type="natural", **unused):
        # x: (nx,)
        xshape = x.shape
        nx = xshape[-1]

        spline_mat = _get_spline_mat_inv(x, bc_type=bc_type)
        self.spline_mat = spline_mat  # (nx, nx)
        self.xshape = xshape
        self.wy = get_trapz_weights(x)  # (nx, nx)
        self.wk = get_cspline_grad_weights(x)  # (nx, nx)

    def cumsum(self, y):
        # y: (*, nx)
        # return: (*, nx)
        y1 = y.unsqueeze(-1)  # (*, nx, 1)
        ks = torch.matmul(self.spline_mat, y1)  # (*, nx, 1)
        kfactor = torch.matmul(self.wk, ks)  # (*, nx, 1)
        yfactor = torch.matmul(self.wy, y1)  # (*, nx, 1)
        res = kfactor + yfactor  # (*, nx)
        return res.squeeze(-1)

    def integrate(self, y):
        ks = torch.matmul(self.spline_mat, y.unsqueeze(-1)).squeeze(-1)  # (*, nx)
        kfactor = torch.einsum("c,...c->...", self.wk[-1], ks)
        yfactor = torch.einsum("c,...c->...", self.wy[-1], y)
        return kfactor + yfactor

    def getparamnames(self, methodname, prefix=""):
        if methodname == "cumsum" or methodname == "integrate":
            return [prefix + "spline_mat", prefix + "wk", prefix + "wy"]
        else:
            raise KeyError("%s has no %s method" % (self.__class__.__name__, methodname))

######################## weight-based cumsum quadrature ########################
class WeightBasedSQuad(BaseSQuad):
    def __init__(self, x, **options):
        # x: (nx,)
        xshape = x.shape
        nx = xshape[-1]
        x = x.reshape(-1, nx)
        self.w = self.get_weights(x, **options)  # (*, nx, nx)

    @abstractmethod
    def get_weights(self, x, **options):
        pass

    def cumsum(self, y):
        # y: (*, nx)
        # w: (*, nx, nx)
        # returns: (*, nx)
        return torch.sum(y.unsqueeze(-2) * self.w, dim=-1)

    def integrate(self, y):
        # y: (*, nx)
        # w: (*, nx, nx)
        # returns: (*,)
        return torch.sum(y * self.w[..., -1, :], dim=-1)

    def getparamnames(self, methodname, prefix=""):
        if methodname == "cumsum" or methodname == "integrate":
            return [prefix + "w"]
        else:
            raise KeyError("%s has no %s method" % (self.__class__.__name__, methodname))

class TrapzSQuad(WeightBasedSQuad):
    r"""
    Perform integration with trapezoidal rule. It is simply

    .. math::

        S = \sum_{i=0}^{N-2} \frac{1}{2}(y_i+y_{i+1})
    """

    def get_weights(self, x):
        return get_trapz_weights(x)

class SimpsonSQuad(WeightBasedSQuad):
    """
    Perform integration with composite Simpson's rule
    """

    def get_weights(self, x):
        return get_simpson_weights(x)

# @torch.jit.script
def get_trapz_weights(x: torch.Tensor) -> torch.Tensor:
    # x: (..., nx)
    # returns: (..., nx, nx)
    half_dx = (x[..., 1:] - x[..., :-1]) * 0.5  # (..., nx-1)
    nx = x.shape[-1]
    shape = list(x.shape[:-1]) + [nx, nx]
    res = torch.zeros(shape, dtype=x.dtype, device=x.device)
    for i in range(1, nx):
        res[..., i:, i - 1:i + 1] += half_dx[..., i - 1:i].unsqueeze(-1)
    return res

# @torch.jit.script
def get_simpson_weights(x: torch.Tensor) -> torch.Tensor:
    # x: (..., nx)
    # returns: (..., nx, nx)
    # ref: https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    h = x[..., 1:] - x[..., :-1]  # (..., nx-1)
    h1 = h[..., 1::2]  # (..., (nx-2)//2)
    h0 = h[..., :-1:2]  # (..., (nx-2)//2)
    h1_2 = h1 * h1
    h0_2 = h0 * h0
    h1_3 = h1_2 * h1
    h0_3 = h0_2 * h0
    alpha = (2 * h1_3 - h0_3 + 3 * h0 * h1_2) / (6 * h1 * (h1 + h0))  # (..., (nx-2)//2)
    eta   = (2 * h0_3 - h1_3 + 3 * h1 * h0_2) / (6 * h0 * (h1 + h0))  # (..., (nx-2)//2)
    beta  = (h1_3 + h0_3 + 3 * h1 * h0 * (h1 + h0)) / (6 * h1 * h0)  # (..., (nx-2)//2)
    # last part (for odd parts only)
    hN1 = h[..., 2::2]  # (..., (nx-3)//2)
    hN2 = h[..., 1:-1:2]  # (..., (nx-3)//2)
    alpha_l = (2 * hN1 * hN1 + 3 * hN1 * hN2) / (6 * (hN1 + hN2))
    eta_l   = hN1 * hN1 * hN1 / (6 * hN2 * (hN1 + hN2))
    beta_l  = (hN1 * hN1 + 3 * hN1 * hN2) / (6 * hN2)

    nx = x.shape[-1]
    shape = list(x.shape[:-1]) + [nx, nx]
    res = torch.zeros(shape, dtype=x.dtype, device=x.device)
    for i in range(2, nx, 2):
        j = i // 2 - 1
        res[..., i:, i - 2] += eta[..., j:j + 1]
        res[..., i:, i - 1] += beta[..., j:j + 1]
        res[..., i:, i] += alpha[..., j:j + 1]
    for i in range(3, nx, 2):  # last part of the odd parts
        j = i // 2 - 1
        res[..., i, i - 2] += -eta_l[..., j]
        res[..., i, i - 1] += beta_l[..., j]
        res[..., i, i] += alpha_l[..., j]

    # trapezoidal rule for the part with N=1 interval
    res[..., 1, :2] = 0.5 * h[..., 0]

    return res

# @torch.jit.script
def get_cspline_grad_weights(x):
    # x: (..., nx)
    # returns: (..., nx, nx)
    dx = (x[..., 1:] - x[..., :-1])  # (..., nx-1)
    dx_factor = dx * dx / 12.  # (..., nx-1)
    sign = torch.tensor([1., -1.], dtype=x.dtype, device=x.device)
    nx = x.shape[-1]
    shape = list(x.shape[:-1]) + [nx, nx]
    res = torch.zeros(shape, dtype=x.dtype, device=x.device)
    for i in range(1, nx):
        res[..., i:, i - 1:i + 1] += dx_factor[..., i - 1:i].unsqueeze(-1) * sign
    return res
