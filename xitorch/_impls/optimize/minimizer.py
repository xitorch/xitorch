import torch
from typing import Callable, List

def gd(fcn: Callable[..., torch.Tensor], x0: torch.Tensor, params: List,
       # gd parameters
       step: float = 1e-3,
       gamma: float = 0.9,
       # stopping conditions
       maxiter: int = 1000,
       f_tol: float = 0.0,
       f_rtol: float = 1e-8,
       x_tol: float = 0.0,
       x_rtol: float = 1e-8,
       # misc parameters
       verbose=False,
       **unused):
    r"""
    Vanilla gradient descent with momentum. The stopping conditions use OR criteria.
    The update step is following the equations below.

    .. math::
        \mathbf{v}_{t+1} &= \gamma \mathbf{v}_t - \eta \nabla_{\mathbf{x}} f(\mathbf{x}_t) \\
        \mathbf{x}_{t+1} &= \mathbf{x}_t + \mathbf{v}_{t+1}

    Keyword arguments
    -----------------
    step: float
        The step size towards the steepest descent direction, i.e. :math:`\eta` in
        the equations above.
    gamma: float
        The momentum factor, i.e. :math:`\gamma` in the equations above.
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
    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, verbose)
    fprev = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    v = torch.zeros_like(x)
    for i in range(maxiter):
        f, dfdx = fcn(x, *params)

        # update the step
        v = (gamma * v - step * dfdx).detach()
        xprev = x.detach()
        x = (xprev + v).detach()

        # check the stopping conditions
        to_stop = stop_cond.to_stop(i, x, xprev, f, fprev)

        if to_stop:
            break

        fprev = f
    return x

def adam(fcn: Callable[..., torch.Tensor], x0: torch.Tensor, params: List,
         # gd parameters
         step: float = 1e-3,
         beta1: float = 0.9,
         beta2: float = 0.999,
         eps: float = 1e-8,
         # stopping conditions
         maxiter: int = 1000,
         f_tol: float = 0.0,
         f_rtol: float = 1e-8,
         x_tol: float = 0.0,
         x_rtol: float = 1e-8,
         # misc parameters
         verbose=False,
         **unused):
    r"""
    Adam optimizer by Kingma & Ba (2015). The stopping conditions use OR criteria.
    The update step is following the equations below.

    .. math::
        \mathbf{g}_t &= \nabla_{\mathbf{x}} f(\mathbf{x}_{t-1}) \\
        \mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \\
        \mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \\
        \hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t) \\
        \hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t) \\
        \mathbf{x}_t &= \mathbf{x}_{t-1} - \alpha \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)

    Keyword arguments
    -----------------
    step: float
        The step size towards the descent direction, i.e. :math:`\alpha` in
        the equations above.
    beta1: float
        Exponential decay rate for the first moment estimate.
    beta2: float
        Exponential decay rate for the first moment estimate.
    eps: float
        Small number to prevent division by 0.
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
    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, verbose)
    fprev = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    v = torch.zeros_like(x)
    m = torch.zeros_like(x)
    beta1t = beta1
    beta2t = beta2
    for i in range(maxiter):
        f, dfdx = fcn(x, *params)
        f = f.detach()
        dfdx = dfdx.detach()

        # update the step
        m = beta1 * m + (1 - beta1) * dfdx
        v = beta2 * v + (1 - beta2) * dfdx ** 2
        mhat = m / (1 - beta1t)
        vhat = v / (1 - beta2t)
        beta1t *= beta1
        beta2t *= beta2
        xprev = x.detach()
        x = (xprev - step * mhat / (vhat ** 0.5 + eps)).detach()

        # check the stopping conditions
        to_stop = stop_cond.to_stop(i, x, xprev, f, fprev)

        if to_stop:
            break

        fprev = f
    return x

class TerminationCondition(object):
    def __init__(self, f_tol: float, f_rtol: float, x_tol: float, x_rtol: float,
                 verbose: bool):
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.verbose = verbose

    def to_stop(self, i: int, x: torch.Tensor, xprev: torch.Tensor,
                f: torch.Tensor, fprev: torch.Tensor) -> bool:
        xnorm = float(x.detach().norm())
        dxnorm = float((xprev - x).detach().norm())
        f = float(f.detach())
        df = float((fprev - f).detach().abs())

        xtcheck = dxnorm < self.x_tol
        xrcheck = dxnorm < self.x_rtol * xnorm
        ytcheck = df < self.f_tol
        yrcheck = df < self.f_rtol * f
        converge = xtcheck or xrcheck or ytcheck or yrcheck
        if self.verbose:
            if i == 0:
                print("   #:             f |        dx,        df")
            if converge:
                print("Finish with convergence")
            if i == 0 or ((i + 1) % 10) == 0 or converge:
                print("%4d: %.6e | %.3e, %.3e" % (i + 1, f, dxnorm, df))
        return i > 0 and converge
