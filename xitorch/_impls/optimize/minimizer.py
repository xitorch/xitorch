import torch
from typing import Callable, List

def gd(fcn: Callable[..., torch.Tensor], x0: torch.Tensor, params: List,
       # gd parameters
       step: float = 1e-3,
       # stopping conditions
       maxiter: int = 1000,
       f_tol: float = 0.0,
       f_rtol: float = 1e-8,
       x_tol: float = 0.0,
       x_rtol: float = 1e-8,
       # misc parameters
       verbose=False,
       **unused):
    """
    Vanilla gradient descent. The stopping conditions use OR criteria.

    Keyword arguments
    -----------------
    step: float
        The step size towards the steepest descent direction.
    maxiter: int
        Maximum number of iterations.
    f_tol: float or None
        The absolute tolerance of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    """

    x = x0.clone()
    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol)
    fprev = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    for i in range(maxiter):
        f, dfdx = fcn(x, *params)
        # f = dfdx.norm()

        # update the step
        xprev = x.detach()
        x = (xprev - step * dfdx).detach()

        if verbose:
            if i == 0 or (i + 1) % 10 == 0:
                print("%4d: %.5e" % (i + 1, float(f.detach())))

        # check the stopping conditions
        if i > 0:
            xnorm = float(x.detach().norm())
            dxnorm = float((xprev - x).detach().norm())
            fnorm = float(f.detach())
            dfnorm = float((fprev - f).detach().abs())
            to_stop = stop_cond.to_stop(xnorm, dxnorm, fnorm, dfnorm)

            if to_stop:
                if verbose:
                    print("Finish with convergence")
                break
        fprev = f
    return x

class TerminationCondition(object):
    def __init__(self, f_tol: float, f_rtol: float, x_tol: float, x_rtol: float):
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol

    def to_stop(self, xnorm: float, dxnorm: float, f: float, df: float) -> bool:
        xtcheck = dxnorm < self.x_tol
        xrcheck = dxnorm < self.x_rtol * xnorm
        ytcheck = f < self.f_tol
        yrcheck = df < self.f_rtol * f
        return xtcheck or xrcheck or ytcheck or yrcheck
