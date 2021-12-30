import torch
import functools
from typing import Optional, Sequence, Callable, Tuple

__all__ = ["rk23_adaptive", "rk45_adaptive"]

def rk_step(func: Callable[..., torch.Tensor], t: torch.Tensor, y: torch.Tensor,
            f: torch.Tensor, h: torch.Tensor,
            abck: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    # t: (...,)
    # y: (..., *ny)
    # f: (..., *ny)
    # h: (...,)
    # A: (norder, norder)
    # B: (norder,)
    # C: (norder,)
    # K: (norder + 1, ..., *ny)
    # ret: (..., *ny), (..., *ny)

    # get the slices to expand from (...,) to (..., *1)
    ylast_slices = (Ellipsis,) + (None,) * (f.ndim - h.ndim)
    ynumel = f.numel()
    yshape = f.shape

    hy = h[ylast_slices]
    A, B, C, K = abck
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = torch.einsum("oy,o->y", K[:s].reshape(-1, ynumel), a[:s]).reshape(yshape) * hy
        K[s] = func(t + c * h, y + dy)
    dydt = torch.einsum("oy,o->y", K[:-1].reshape(-1, ynumel), B).reshape(yshape)
    ynew = y + hy * dydt
    fnew = func(t + h, ynew)
    K[-1] = fnew
    return ynew, fnew

class RKAdaptiveStepSolver(object):
    A: Optional[torch.Tensor] = None
    B: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    E: Optional[torch.Tensor] = None
    n_stages: Optional[int] = None
    error_estimator_order: Optional[int] = None

    def __init__(self, atol: float, rtol: float):
        self.atol = atol
        self.rtol = rtol
        self.max_factor = 10.0
        self.min_factor = 0.2
        self.step_mult = 0.9
        self.error_exponent = -1. / (self.error_estimator_order + 1.)

    def setup(self, fcn: Callable[..., torch.Tensor], ts: torch.Tensor,
              y0: torch.Tensor, params: Sequence[torch.Tensor]):
        # ts: (nt, ...)
        # y0: (..., *ny)
        # fcn_out: (..., *ny)
        self.yshape = y0.shape
        self.ynumel = y0.numel()
        yndims = y0.ndim - ts.ndim + 1
        self.ydims = tuple(range(-1, -yndims - 1, -1))  # the dimension indices of *ny
        self.y0 = y0

        # get the slices to expand from (...,) to (..., *1)
        ylast_slices = (Ellipsis,) + (None,) * yndims
        self.ylast_slices = ylast_slices

        direction = ts[1] - ts[0]  # (...,)
        sgn = torch.sign(direction)  # (...,)
        sgny = sgn[ylast_slices]  # (..., *1)
        self.ts = sgn * ts  # (nt, ...)
        self.func = lambda t, y: sgny * fcn(sgn * t, y, *params)
        self.dtype = y0.dtype
        self.device = y0.device
        # K: (norder + 1, ..., *ny)
        self.K = torch.empty((self.n_stages + 1,) + y0.shape, dtype=self.dtype, device=self.device)

        # convert the predefined tensors into the dtype and device
        self.A = self.A.to(self.dtype).to(self.device)
        self.B = self.B.to(self.dtype).to(self.device)
        self.C = self.C.to(self.dtype).to(self.device)
        self.E = self.E.to(self.dtype).to(self.device)

    def solve(self) -> torch.Tensor:
        # ts: (nt, ...)
        # y0: (..., *ny)
        # yt: (nt, ..., *ny)
        t0 = self.ts[0]
        ts = self.ts
        f0 = self.func(t0, self.y0)
        h0 = self.ts[1] - self.ts[0]  # TODO: perform more intelligent guess

        # prepare the results
        nt = len(ts)
        yt = torch.empty((len(self.ts),) + self.yshape, dtype=self.dtype, device=self.device)
        yt[0] = self.y0

        rk_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        rk_state = (f0, t0, self.y0, h0)
        for i in range(1, len(ts)):
            rk_state = self._step(rk_state, ts[i])
            yt[i] = rk_state[2]
        return yt

    def _error_norm(self, K: torch.Tensor, h: torch.Tensor):
        # K: (norder + 1, ..., *ny)
        # E: (norder + 1)
        # h: (...,)

        # err: (..., *ny)
        err = torch.einsum("oy,o->y", K.reshape(-1, self.ynumel), self.E).reshape(self.yshape)
        err = err * h[self.ylast_slices]
        # returns: (...,)
        res = err.norm(dim=self.ydims)
        return res

    def _step(self, rk_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor ,torch.Tensor],
              t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t1_achieved = False
        while not t1_achieved:
            rk_state, t1_achieved = self._single_step(rk_state, t1)
        return rk_state

    def _single_step(self, rk_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     t1: torch.Tensor) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]:

        f0, t0, y0, h = rk_state
        accepted = False
        prev_rejected = False
        while not accepted:
            # check if the current step exceeds the target
            t1_achieved = t0 + h > t1  # (...,)
            hstep = torch.where(t1_achieved, t1 - t0, h)  # (...,)
            tnew = t0 + hstep

            # perform the RK-step to t0+h
            abck = (self.A, self.B, self.C, self.K)
            ynew, fnew = rk_step(self.func, t0, y0, f0, hstep, abck)

            # estimate the error norm
            scale = self.atol + torch.maximum(y0.norm(dim=self.ydims), ynew.norm(dim=self.ydims)) * self.rtol
            errnorm = self._error_norm(self.K, hstep) / scale  # (...,)
            accepted = bool(torch.all(errnorm < 1).cpu().detach().item())

            # adjust the step size
            new_factor = self.step_mult * (errnorm + 1e-8) ** self.error_exponent
            # if accepted and not t1_achieved:
            if accepted:
                factor = torch.minimum(torch.full_like(new_factor, self.max_factor), new_factor)
                if prev_rejected:
                    factor = torch.minimum(torch.ones_like(factor), factor)
                not_t1_achieved = torch.logical_not(t1_achieved)
                h[not_t1_achieved] *= factor[not_t1_achieved]

            else:
                factor = torch.maximum(torch.full_like(new_factor, self.min_factor), new_factor)
                h = hstep * factor

            prev_rejected = not accepted

        rk_state = (fnew, tnew, ynew, h)
        t1_achieved_all = bool(torch.all(t1_achieved).cpu().detach().item())
        return rk_state, t1_achieved_all

class RK23(RKAdaptiveStepSolver):
    error_estimator_order = 2
    n_stages = 3
    C = torch.tensor([0, 1 / 2, 3 / 4], dtype=torch.float64)
    A = torch.tensor([
        [0, 0, 0],
        [1 / 2, 0, 0],
        [0, 3 / 4, 0]
    ], dtype=torch.float64)
    B = torch.tensor([2 / 9, 1 / 3, 4 / 9], dtype=torch.float64)
    E = torch.tensor([5 / 72, -1 / 12, -1 / 9, 1 / 8], dtype=torch.float64)

class RK45(RKAdaptiveStepSolver):
    error_estimator_order = 4
    n_stages = 6
    C = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1], dtype=torch.float64)
    A = torch.tensor([
        [0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
    ], dtype=torch.float64)
    B = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=torch.float64)
    E = torch.tensor([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525,
                      1 / 40], dtype=torch.float64)

def _rk_adaptive(fcn: Callable[..., torch.Tensor], ts: torch.Tensor,
                 y0: torch.Tensor, params: Sequence[torch.Tensor],
                 cls, atol: float = 1e-8, rtol: float = 1e-5, **unused):
    """
    Keyword arguments
    -----------------
    atol: float
        The absolute error tolerance in deciding the steps
    rtol: float
        The relative error tolerance in deciding the steps
    """
    solver = cls(atol=atol, rtol=rtol)
    solver.setup(fcn, ts, y0, params)
    return solver.solve()

@functools.wraps(_rk_adaptive, assigned='__annotations__')
def rk23_adaptive(fcn: Callable[..., torch.Tensor], ts: torch.Tensor,
                  y0: torch.Tensor, params: Sequence[torch.Tensor], **kwargs):
    """
    Perform the adaptive Runge-Kutta steps with order 2 and 3.
    """
    return _rk_adaptive(fcn, ts, y0, params, RK23, **kwargs)

@functools.wraps(_rk_adaptive, assigned='__annotations__')
def rk45_adaptive(fcn: Callable[..., torch.Tensor], ts: torch.Tensor,
                  y0: torch.Tensor, params: Sequence[torch.Tensor], **kwargs):
    """
    Perform the adaptive Runge-Kutta steps with order 4 and 5.
    """
    return _rk_adaptive(fcn, ts, y0, params, RK45, **kwargs)


# complete the docstring
rk23_adaptive.__doc__ += _rk_adaptive.__doc__  # type: ignore
rk45_adaptive.__doc__ += _rk_adaptive.__doc__  # type: ignore
