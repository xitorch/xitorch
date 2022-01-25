from typing import Optional
import torch
import warnings
from abc import abstractmethod
from xitorch._impls.interpolate.base_interp import BaseInterp
from xitorch._impls.interpolate.extrap_utils import get_extrap_pos, get_extrap_val
from xitorch._utils.bcast import match_dim

class BaseInterp1D(BaseInterp):
    def __init__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, extrap: Optional[str] = None,
                 **unused):
        # x: (*BX, nr)
        # y: (*BY, nr), BX and BY are broadcastable
        self._y_is_given = y is not None
        self._extrap = extrap
        self._xmin = torch.min(x, dim=-1, keepdim=True)[0]
        self._xmax = torch.max(x, dim=-1, keepdim=True)[0]
        self._is_periodic_required = False
        self._y = y

    def set_periodic_required(self, val):
        self._is_periodic_required = val

    def is_periodic_required(self):
        return self._is_periodic_required

    def __call__(self, xq: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # xq: (*BX, nrq)
        # y: (*BY, nr)
        if self._y_is_given and y is not None:
            msg = "y has been supplied when initiating this instance. This value of y will be ignored"
            # stacklevel=3 because this __call__ will be called by a wrapper's __call__
            warnings.warn(msg, stacklevel=3)

        extrap = self._extrap
        if self._y_is_given:
            y = self._y
        elif y is None:
            raise RuntimeError("y must be given")
        elif self.is_periodic_required():
            check_periodic_value(y)
        assert y is not None

        xqinterp_mask = torch.logical_and(xq >= self._xmin, xq <= self._xmax)  # (*BX, nrq)
        xqextrap_mask = ~xqinterp_mask
        allinterp = torch.all(xqinterp_mask)

        if not allinterp and xqextrap_mask.ndim > 1:
            raise NotImplementedError("Batched interpolation + extrapolation has not been implemented yet")

        if allinterp:
            return self._interp(xq, y=y)
        elif extrap == "mirror" or extrap == "periodic" or extrap == "bound":
            # extrapolation by mapping it to the interpolated region
            xq2 = xq.clone()
            xq2[xqextrap_mask] = get_extrap_pos(xq[xqextrap_mask], extrap, self._xmin, self._xmax)
            return self._interp(xq2, y=y)
        else:
            # interpolation
            yqinterp = self._interp(xq[xqinterp_mask], y=y)  # (*BY, nrq)
            yqextrap = get_extrap_val(xq[xqextrap_mask], y, extrap)

            yq = torch.empty((*y.shape[:-1], xq.shape[-1]), dtype=y.dtype, device=y.device)  # (*BY, nrq)
            yq[..., xqinterp_mask] = yqinterp
            yq[..., xqextrap_mask] = yqextrap
            return yq

    @abstractmethod
    def _interp(self, xq: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

class CubicSpline1D(BaseInterp1D):
    def __init__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                 bc_type: Optional[str] = None, extrap: Optional[str] = None, **unused):
        # x: (*BX, nr)
        # y: (*BY, nr), BX and BY are broadcastable

        # get the default extrapolation method and boundary condition
        if bc_type is None:
            bc_type = "not-a-knot"
        extrap = check_and_get_extrap(extrap, bc_type)
        super(CubicSpline1D, self).__init__(x, y, extrap=extrap)

        self.x = x
        # if x.ndim != 1:
        #     raise RuntimeError("The input x must be a 1D tensor")

        bc_types = ["natural", "clamped", "not-a-knot", "periodic"]
        if bc_type not in bc_types:
            raise RuntimeError("Unimplemented %s bc_type. Available options: %s" % (bc_type, bc_types))
        self.bc_type = bc_type
        self.set_periodic_required(extrap == "periodic")  # or self.bc_type == "periodic"

        # precompute the inverse of spline matrix
        self.spline_mat_inv = _get_spline_mat_inv(x, bc_type)  # (*BX, nr, nr)
        self.y_is_given = y is not None
        if self.y_is_given:
            if self.is_periodic_required():
                check_periodic_value(y)
            self.y = y
            assert y is not None
            self.ks = torch.matmul(self.spline_mat_inv, y.unsqueeze(-1)).squeeze(-1)

    def _interp(self, xq, y):
        # https://en.wikipedia.org/wiki/Spline_interpolation#Algorithm_to_find_the_interpolating_cubic_spline
        # get the k-vector (i.e. the gradient at every points)
        if self.y_is_given:
            ks = self.ks
        else:
            ks = torch.matmul(self.spline_mat_inv, y.unsqueeze(-1)).squeeze(-1)  # (*BY, nr)

        x, xq = match_dim(self.x, xq, contiguous=True)  # (*BX, nr)

        # find the index location of xq
        nr = x.shape[-1]
        # detaching due to PyTorch's issue #42328
        idxr = torch.searchsorted(x.detach(), xq.detach(), right=False)  # (*BX, nrq)
        idxr = torch.clamp(idxr, 1, nr - 1)
        idxl = idxr - 1  # (*BX, nrq) from (0 to nr-2)

        if torch.numel(xq) > torch.numel(x):
            # get the variables needed
            yl = y[..., :-1]  # (*BY, nr-1)
            xl = x[..., :-1]  # (nr-1)
            dy = y[..., 1:] - yl  # (*BY, nr-1)
            dx = x[..., 1:] - xl  # (nr-1)
            a = ks[..., :-1] * dx - dy  # (*BY, nr-1)
            b = -ks[..., 1:] * dx + dy  # (*BY, nr-1)

            # calculate the coefficients for the t-polynomial
            p0 = yl  # (*BY, nr-1)
            p1 = (dy + a)  # (*BY, nr-1)
            p2 = (b - 2 * a)  # (*BY, nr-1)
            p3 = a - b  # (*BY, nr-1)

            t = (xq - torch.gather(xl, -1, idxl)) / torch.gather(dx, -1, idxl)  # (*BX, nrq)
            # yq = p0[:,idxl] + t * (p1[:,idxl] + t * (p2[:,idxl] + t * p3[:,idxl])) # (nbatch, nrq)
            p0, p1, p2, p3, idxl = match_dim(p0, p1, p2, p3, idxl)
            yq = torch.gather(p3, dim=-1, index=idxl) * t
            yq += torch.gather(p2, dim=-1, index=idxl)
            yq *= t
            yq += torch.gather(p1, dim=-1, index=idxl)
            yq *= t
            yq += torch.gather(p0, dim=-1, index=idxl)
            return yq

        else:
            x, y, ks, idxl, idxr = match_dim(x, y, ks, idxl, idxr)
            xl = torch.gather(x, dim=-1, index=idxl)
            xr = torch.gather(x, dim=-1, index=idxr)
            yl = torch.gather(y, dim=-1, index=idxl).contiguous()
            yr = torch.gather(y, dim=-1, index=idxr).contiguous()
            kl = torch.gather(ks, dim=-1, index=idxl).contiguous()
            kr = torch.gather(ks, dim=-1, index=idxr).contiguous()

            dxrl = xr - xl  # (nrq,)
            # dyrl = yr - yl  # (nbatch, nrq)

            # calculate the coefficients of the large matrices
            t = (xq - xl) / dxrl  # (nrq,)
            tinv = 1 - t  # nrq
            tta = t * tinv * tinv
            ttb = t * tinv * t
            tyl = tinv + tta - ttb
            tyr = t - tta + ttb
            tkl = tta * dxrl
            tkr = -ttb * dxrl

            yq = yl * tyl + yr * tyr + kl * tkl + kr * tkr
            return yq

    def getparamnames(self):
        if self.y_is_given:
            res = ["x", "y", "ks"]
        else:
            res = ["spline_mat_inv", "x"]
        return res

class LinearInterp1D(BaseInterp1D):
    def __init__(self, x, y=None, extrap=None, **unused):
        super(LinearInterp1D, self).__init__(x, y, extrap=extrap)
        self.x = x
        self.y_is_given = y is not None
        self.y = y

    def _interp(self, xq: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.y_is_given:
            y = self.y

        x, xq = match_dim(self.x, xq, contiguous=True)

        nr = x.shape[-1]
        idxr = torch.searchsorted(x.detach(), xq.detach(), right=False)  # (nrq)
        idxr = torch.clamp(idxr, 1, nr - 1)
        idxl = idxr - 1  # (nrq) from (0 to nr-2)

        if torch.numel(xq) > torch.numel(x):
            x, y, idxl = match_dim(x, y, idxl)
            yl = y[..., :-1]  # (..., nr-1)
            xl = x[..., :-1]  # (..., nr-1)
            dy = y[..., 1:] - yl  # (..., nr-1)
            dx = x[..., 1:] - xl  # (..., nr-1)
            t = (xq - torch.gather(xl, -1, idxl)) / torch.gather(dx, -1, idxl)  # (..., nrq)
            yq = torch.gather(dy, dim=-1, index=idxl) * t
            yq += torch.gather(yl, dim=-1, index=idxl)
            return yq
        else:
            x, y, idxl, idxr = match_dim(x, y, idxl, idxr)
            xl = torch.gather(x, -1, idxl)
            xr = torch.gather(x, -1, idxr)
            yl = torch.gather(y, dim=-1, index=idxl).contiguous()
            yr = torch.gather(y, dim=-1, index=idxr).contiguous()

            dxrl = xr - xl  # (..., nrq)
            dyrl = yr - yl  # (..., nrq)
            t = (xq - xl) / dxrl  # (..., nrq)
            yq = yl + dyrl * t
            return yq

    def getparamnames(self):
        if self.y_is_given:
            res = ["x", "y"]
        else:
            res = ["x"]
        return res

##### docstrings #####
extrap_docstr = """
    extrap: int, float, 1-element torch.Tensor, str, or None
        Extrapolation option:

        * ``int``, ``float``, or 1-element ``torch.Tensor``: it will pad the extrapolated
          values with the specified values
        * ``"mirror"``: the extrapolation values are mirrored
        * ``"periodic"``: periodic boundary condition. ``y[...,0] == y[...,-1]`` must
          be fulfilled for this condition.
        * ``"bound"``: fill in the extrapolated values with the left or right bound
          values.
        * ``"nan"``: fill the extrapolated values with nan
        * callable: apply this extrapolation function with the extrapolated
          positions and use the output as the values
        * ``None``: choose the extrapolation based on the ``bc_type``. These are the
          pairs:

          * ``"clamped"``: ``"mirror"``
          * other: ``"nan"``

        Default: ``None``"""

CubicSpline1D.__doc__ = """
    Perform 1D cubic spline interpolation for non-uniform ``x`` [1]_ [2]_.

    Keyword arguments
    -----------------
    bc_type: str or None
        Boundary condition:

        * ``"not-a-knot"``: The first and second segments are the same polynomial
        * ``"natural"``: 2nd grad at the boundaries are 0
        * ``"clamped"``: 1st grad at the boundaries are 0
        * ``"periodic"``: periodic boundary condition (`new in version 0.2`)

        If ``None``, it will choose ``"not-a-knot"``
    """ + extrap_docstr + """

    References
    ----------
    .. [1] SplineInterpolation on Wikipedia,
           https://en.wikipedia.org/wiki/Spline_interpolation#Algorithm_to_find_the_interpolating_cubic_spline)
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
"""

LinearInterp1D.__doc__ = """
    Perform 1D linear interpolation for non-uniform ``x``.

    Keyword arguments
    -----------------""" + extrap_docstr + """
"""

def check_and_get_extrap(extrap, bc_type):
    if extrap is None:
        try:
            return {
                "clamped": "mirror",
                "periodic": "periodic",
            }[bc_type]
        except KeyError:
            return "nan"
    return extrap

def check_periodic_value(y):
    if not torch.allclose(y[..., 0], y[..., -1]):
        raise RuntimeError("The value of y must be periodic to have periodic bc_type or extrap")

# @torch.jit.script
def _get_spline_mat_inv(x: torch.Tensor, bc_type: str):
    """
    Returns the inverse of spline matrix where the gradient can be obtained just
    by

    >>> spline_mat_inv = _get_spline_mat_inv(x, transpose=True)
    >>> ks = torch.matmul(y, spline_mat_inv)

    where `y` is a tensor of (nbatch, nr) and `spline_mat_inv` is the output of
    this function with shape (nr, nr)

    Arguments
    ---------
    x: torch.Tensor with shape (*BX, nr)
        The x-position of the data
    bc_type: str
        The boundary condition

    Returns
    -------
    mat: torch.Tensor with shape (*BX, nr, nr)
        The inverse of spline matrix.
    """
    nr = x.shape[-1]
    BX = x.shape[:-1]
    matshape = (*BX, nr, nr)

    device = x.device
    dtype = x.dtype

    # construct the matrix for the left hand side
    dxinv0 = 1. / (x[..., 1:] - x[..., :-1])  # (*BX,nr-1)
    zero_pad = torch.zeros_like(dxinv0[..., :1])
    dxinv = torch.cat((zero_pad, dxinv0, zero_pad), dim=-1)
    diag = (dxinv[..., :-1] + dxinv[..., 1:]) * 2  # (*BX,nr)
    offdiag = dxinv0  # (*BX,nr-1)
    spline_mat = torch.zeros(matshape, dtype=dtype, device=device)
    spdiag = spline_mat.diagonal(dim1=-2, dim2=-1)  # (*BX, nr)
    spudiag = spline_mat.diagonal(offset=1, dim1=-2, dim2=-1)
    spldiag = spline_mat.diagonal(offset=-1, dim1=-2, dim2=-1)
    spdiag[..., :] = diag
    spudiag[..., :] = offdiag
    spldiag[..., :] = offdiag

    # construct the matrix on the right hand side
    dxinv2 = (dxinv * dxinv) * 3
    diagr = (dxinv2[..., :-1] - dxinv2[..., 1:])
    udiagr = dxinv2[..., 1:-1]
    ldiagr = -udiagr
    matr = torch.zeros(matshape, dtype=dtype, device=device)
    matrdiag = matr.diagonal(dim1=-2, dim2=-1)
    matrudiag = matr.diagonal(offset=1, dim1=-2, dim2=-1)
    matrldiag = matr.diagonal(offset=-1, dim1=-2, dim2=-1)
    matrdiag[..., :] = diagr
    matrudiag[..., :] = udiagr
    matrldiag[..., :] = ldiagr

    # modify the matrices according to the boundary conditions
    if bc_type == "natural":
        pass  # set to be natural
    elif bc_type == "clamped":
        spline_mat[..., 0, :] = 0.
        spline_mat[..., 0, 0] = 1.
        spline_mat[..., -1, :] = 0.
        spline_mat[..., -1, -1] = 1.
        matr[..., 0, :] = 0.
        matr[..., -1, :] = 0.
    elif bc_type == "not-a-knot":
        dxinv00_sq = dxinv0[..., 0]**2
        dxinv01_sq = dxinv0[..., 1]**2
        dxinv0n_sq = dxinv0[..., -1]**2
        dxinv0nm1_sq = dxinv0[..., -2]**2
        dxinv00_3 = dxinv0[..., 0] * dxinv00_sq
        dxinv01_3 = dxinv0[..., 1] * dxinv01_sq
        dxinv0n_3 = dxinv0[..., -1] * dxinv0n_sq
        dxinv0nm1_3 = dxinv0[..., -2] * dxinv0nm1_sq
        spline_mat[..., 0, 0] = dxinv00_sq
        spline_mat[..., 0, 1] = dxinv00_sq - dxinv01_sq
        spline_mat[..., 0, 2] = -dxinv01_sq
        spline_mat[..., -1, -1] = -dxinv0n_sq
        spline_mat[..., -1, -2] = dxinv0nm1_sq - dxinv0n_sq
        spline_mat[..., -1, -3] = dxinv0nm1_sq
        matr[..., 0, 0] = 2 * (-dxinv00_3)
        matr[..., 0, 1] = 2 * (dxinv00_3 + dxinv01_3)
        matr[..., 0, 2] = 2 * (-dxinv01_3)
        matr[..., -1, -1] = 2 * (-dxinv0n_3)
        matr[..., -1, -2] = 2 * (dxinv0n_3 + dxinv0nm1_3)
        matr[..., -1, -3] = 2 * (-dxinv0nm1_3)
    elif bc_type == "periodic":
        dxinv01 = dxinv0[..., -1]
        dxinv00 = dxinv0[..., 0]
        spline_mat[..., 0, -2] += dxinv01
        spline_mat[..., 0, 0] += dxinv01 * 2
        spline_mat[..., -1, 1] += dxinv00
        spline_mat[..., -1, -1] += dxinv00 * 2

        dxinv01_sq3 = 3 * dxinv01 * dxinv01
        dxinv00_sq3 = 3 * dxinv00 * dxinv00
        matr[..., 0, -2] -= dxinv01_sq3
        matr[..., 0, 0] += dxinv01_sq3
        matr[..., -1, 1] += dxinv00_sq3
        matr[..., -1, -1] -= dxinv00_sq3
    else:
        raise RuntimeError("Unknown boundary condition: %s" % bc_type)

    # solve the matrix inverse
    spline_mat_inv = torch.linalg.solve(spline_mat, matr)

    # return to the shape of x
    return spline_mat_inv
