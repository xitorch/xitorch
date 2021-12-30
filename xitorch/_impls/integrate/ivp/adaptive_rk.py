import torch
import functools
from typing import Optional

__all__ = ["rk23_adaptive", "rk45_adaptive"]

def rk_step(func, t, y, f, h, abck):
    # A: (norder, norder)
    # B: (norder,)
    # C: (norder,)
    # K: (norder+1, ny)
    A, B, C, K = abck
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = torch.matmul(K[:s].T, a[:s]) * h
        K[s] = func(t + c * h, y + dy)
    ynew = y + h * torch.matmul(K[:-1].T, B)
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

    def __init__(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol
        self.max_factor = 10
        self.min_factor = 0.2
        self.step_mult = 0.9
        self.error_exponent = -1. / (self.error_estimator_order + 1.)

    def setup(self, fcn, ts, y0, params):
        # flatten the y0, will be restore at the end of .solve()
        self.yshape = y0.shape
        self.y0 = y0.reshape(-1)

        direction = ts[1] - ts[0]
        if direction < 0:
            self.ts = -ts
            self.func = lambda t, y: -fcn(-t, y.reshape(self.yshape), *params).reshape(-1)
        else:
            self.ts = ts
            self.func = lambda t, y: fcn(t, y.reshape(self.yshape), *params).reshape(-1)
        self.dtype = y0.dtype
        self.device = y0.device
        n = torch.numel(y0)
        self.K = torch.empty((self.n_stages + 1, n), dtype=self.dtype, device=self.device)

        # convert the predefined tensors into the dtype and device
        self.A = self.A.to(self.dtype).to(self.device)
        self.B = self.B.to(self.dtype).to(self.device)
        self.C = self.C.to(self.dtype).to(self.device)
        self.E = self.E.to(self.dtype).to(self.device)

    def solve(self):
        t0 = self.ts[0]
        ts = self.ts
        f0 = self.func(t0, self.y0)
        h0 = self.ts[1] - self.ts[0]  # ??? perform more intelligent guess

        # prepare the results
        nt = len(ts)
        yt = torch.empty((len(self.ts), *self.y0.shape), dtype=self.dtype, device=self.device)
        yt[0] = self.y0

        rk_state = (f0, t0, self.y0, h0)
        for i in range(1, len(ts)):
            rk_state = self._step(rk_state, ts[i])
            yt[i] = rk_state[2]
        return yt.reshape(-1, *self.yshape)

    def _error_norm(self, K, h):
        err = torch.matmul(K.T, self.E) * h
        return err.norm()

    def _step(self, rk_state, t1):
        t1_achieved = False
        while not t1_achieved:
            rk_state, t1_achieved = self._single_step(rk_state, t1)
        return rk_state

    def _single_step(self, rk_state, t1):
        f0, t0, y0, h = rk_state
        accepted = False
        prev_rejected = False
        while not accepted:
            # check if the current step exceeds the target
            t1_achieved = t0 + h > t1
            hstep = t1 - t0 if t1_achieved else h
            tnew = t0 + hstep

            # perform the RK-step to t0+h
            abck = (self.A, self.B, self.C, self.K)
            ynew, fnew = rk_step(self.func, t0, y0, f0, hstep, abck)

            # estimate the error norm
            scale = self.atol + torch.max(y0.norm(), ynew.norm()) * self.rtol
            errnorm = self._error_norm(self.K, hstep) / scale
            accepted = errnorm < 1

            # adjust the step size
            if accepted and not t1_achieved:
                if errnorm == 0:
                    factor = self.max_factor
                else:
                    factor = min(self.max_factor, self.step_mult * errnorm ** self.error_exponent)

                if prev_rejected:
                    factor = min(1.0, factor)

                h *= factor
            elif not accepted:
                factor = max(self.min_factor, self.step_mult * errnorm ** self.error_exponent)
                h = hstep * factor

            prev_rejected = not accepted

        rk_state = (fnew, tnew, ynew, h)
        return rk_state, t1_achieved

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

def _rk_adaptive(fcn, ts, y0, params, cls, atol=1e-8, rtol=1e-5, **unused):
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
def rk23_adaptive(fcn, ts, y0, params, **kwargs):
    """
    Perform the adaptive Runge-Kutta steps with order 2 and 3.
    """
    return _rk_adaptive(fcn, ts, y0, params, RK23, **kwargs)

@functools.wraps(_rk_adaptive, assigned='__annotations__')
def rk45_adaptive(fcn, ts, y0, params, **kwargs):
    """
    Perform the adaptive Runge-Kutta steps with order 4 and 5.
    """
    return _rk_adaptive(fcn, ts, y0, params, RK45, **kwargs)


# complete the docstring
rk23_adaptive.__doc__ += _rk_adaptive.__doc__  # type: ignore
rk45_adaptive.__doc__ += _rk_adaptive.__doc__  # type: ignore
