# file mostly from SciPy: https://github.com/scipy/scipy/blob/914523af3bc03fe7bf61f621363fca27e97ca1d6/scipy/optimize/nonlin.py#L221
# and converted to PyTorch for GPU efficiency

import torch
import functools
from xitorch._impls.optimize.root._jacobian import BroydenFirst

__all__ = ["broyden1"]

def _nonlin_solver(fcn, x0, params, method,
        # jacobian parameters
        alpha=None, max_rank=None,
        # stopping criteria
        maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
        # algorithm parameters
        line_search=True,
        # misc parameters
        verbose=False,
        **unused):
    """
    Keyword arguments
    -----------------
    alpha: float or None
        The initial guess of Jacobian is ``-1/alpha``
    max_rank: int or None
        The maximum rank of inverse Jacobian approximation. If ``None``, it
        is ``inf``.
    maxiter: int or None
        Maximum number of iterations, or inf if it is set to None.
    f_tol: float or None
        The absolute tolerance of the norm of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the norm of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    line_search: bool or str
        Options to perform line search. If ``True``, it is set to ``"armijo"``.
    verbose: bool
        Options for verbosity
    """

    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol)
    jacobian = {
        "broyden1": BroydenFirst,
    }[method](alpha, max_rank)

    if maxiter is None:
        maxiter = 100*(torch.numel(x0)+1)
    if line_search is True:
        line_search = "armijo"
    elif line_search is False:
        line_search = None

    # shorthand for the function
    xshape = x0.shape
    func = lambda x: fcn(x.reshape(xshape), *params).reshape(-1)
    x = x0.reshape(-1)

    y = func(x)
    y_norm = y.norm()
    if (y_norm == 0):
        return x.reshape(xshape)

    # set up the jacobian
    jacobian.setup(x, y)

    # solver tolerance
    gamma = 0.9
    eta_max = 0.9999
    eta_threshold = 0.1
    eta = 1e-3

    converge = False
    for i in range(maxiter):
        tol = min(eta, eta * y_norm)
        dx = -jacobian.solve(y, tol=tol)

        dx_norm = dx.norm()
        if dx_norm == 0:
            raise ValueError("Jacobian inversion yielded zero vector. "
                             "This indicates a bug in the Jacobian "
                             "approximation.")

        if line_search:
            s, xnew, ynew, y_norm_new = _nonline_line_search(func, x, y, dx,
                search_type=line_search)
        else:
            s = 1.0
            xnew = x + dx
            ynew = func(xnew)
            y_norm_new = ynew.norm()

        jacobian.update(xnew.clone(), ynew)

        # print out dx and df
        dy = (ynew - y)
        to_stop = stop_cond.check(x.norm(), y_norm, dx_norm, dy.norm())
        if verbose:
            if i < 10 or i % 10 == 0 or to_stop:
                print("%6d: |dx|=%.3e, |df|=%.3e" % (i, dx_norm, dy.norm()))
        if to_stop:
            converge = True
            break

        # adjust forcing parameters for inexact solve
        eta_A = float(gamma * (y_norm_new / y_norm)**2)
        gamma_eta2 = gamma * eta * eta
        if gamma_eta2 < eta_threshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma_eta2))

        y_norm = y_norm_new
        x = xnew
        y = ynew
    if not converge:
        msg = "The rootfinder does not converge after %d iterations. " + \
              "|dx|=%.3e, |df|=%.3e"
        warnings.warn(msg % (maxiter, dx_norm, dy.norm()))
    return x.reshape(xshape)

@functools.wraps(_nonlin_solver, assigned=('__annotations__',)) # takes only the signature
def broyden1(fcn, x0, params=(), **kwargs):
    """
    Solve the root finder or linear equation using the first Broyden method [1]_.
    It can be used to solve minimization by finding the root of the
    function's gradient.

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional systems of nonlinear equations".
           Mathematisch Instituut, Universiteit Leiden, The Netherlands (2003).
           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf
    """
    return _nonlin_solver(fcn, x0, params, "broyden1", **kwargs)

# set the docstring of the functions
broyden1.__doc__ += _nonlin_solver.__doc__

def _safe_norm(v):
    if not torch.isfinite(v).all():
        return torch.tensor(float("inf"), dtype=v.dtype, device=v.device)
    return torch.norm(v)

def _nonline_line_search(func, x, y, dx, search_type="armijo", rdiff=1e-8, smin=1e-2):
    tmp_s = [0]
    tmp_y = [y]
    tmp_phi = [y.norm()**2]
    s_norm = x.norm() / dx.norm()

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]
        xt = x + s*dx
        v = func(xt)
        p = _safe_norm(v)**2
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_y[0] = v
        return p

    def derphi(s):
        ds = (torch.abs(s) + s_norm + 1) * rdiff
        return (phi(s+ds, store=False) - phi(s)) / ds

    if search_type == 'armijo':
        s, phi1 = _scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0],
                                        amin=smin)

    if s is None:
        # No suitable step length found. Take the full Newton step,
        # and hope for the best.
        s = 1.0

    x = x + s*dx
    if s == tmp_s[0]:
        y = tmp_y[0]
    else:
        y = func(x)
    y_norm = y.norm()

    return s, x, y, y_norm

def _scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0, max_niter=20):
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    niter = 0
    while alpha1 > amin and niter < max_niter:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
        niter += 1

    # Failed to find a suitable step length
    if niter == max_niter:
        return alpha2, phi_a2
    return None, phi_a1

class TerminationCondition(object):
    def __init__(self, f_tol, f_rtol, x_tol, x_rtol):
        if f_tol is None:
            f_tol = 1e-6
        if f_rtol is None:
            f_rtol = float("inf")
        if x_tol is None:
            x_tol = 1e-6
        if x_rtol is None:
            x_rtol = float("inf")
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol

    def check(self, xnorm, ynorm, dxnorm, dynorm):
        xtcheck = dxnorm < self.x_tol
        xrcheck = dxnorm < self.x_rtol * xnorm
        ytcheck = dynorm < self.f_tol
        yrcheck = dynorm < self.f_rtol * ynorm
        return xtcheck and xrcheck and ytcheck and yrcheck
