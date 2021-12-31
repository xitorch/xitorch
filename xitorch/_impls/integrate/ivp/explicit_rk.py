from typing import List, Callable, Sequence, NamedTuple, Union
import torch

# All functions in this file should have the following inputs and outputs
# Inputs
# ------
# * fcn: callable dy/dt = fcn(t, y, *params)
#       The function to be integrated. It should produce output of list of
#       tensors following the shapes of tuple `y`. `t` should be a single element.
# * t: torch.Tensor (nt,)
#       The integrated times
# * y0: list of torch.Tensor (*ny)
#       The list of initial values
# * params: list
#       List of any other parameters
# * **kwargs: dict
#       Any other keyword arguments
# Outputs
# -------
# * yt: list of torch.Tensor (nt,*ny)
#       The value of `y` at the given time `t`
# Note
# ----
# The operations are done in grad-disabled environment and **not** expected to
# be able to propagate gradients.

__all__ = ["rk4_ivp", "rk38_ivp", "fwd_euler_ivp"]

############################# list of tableaus #############################
class _Tableau(NamedTuple):
    c: List[float]
    b: List[float]
    a: List[List[float]]

rk4_tableau = _Tableau(
    c=[0.0, 0.5, 0.5, 1.0],
    b=[1 / 6., 1 / 3., 1 / 3., 1 / 6.],
    a=[[0.0, 0.0, 0.0, 0.0],
       [0.5, 0.0, 0.0, 0.0],
       [0.0, 0.5, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0]]
)
rk38_tableau = _Tableau(
    c=[0.0, 1 / 3, 2 / 3, 1.0],
    b=[1 / 8, 3 / 8, 3 / 8, 1 / 8],
    a=[[0.0, 0.0, 0.0, 0.0],
       [1 / 3, 0.0, 0.0, 0.0],
       [-1 / 3, 1.0, 0.0, 0.0],
       [1.0, -1.0, 1.0, 0.0]]
)
fwd_euler_tableau = _Tableau(
    c=[0.0],
    b=[1.0],
    a=[[0.0]]
)

def explicit_rk(tableau: _Tableau,
                fcn: Callable[..., torch.Tensor], t: torch.Tensor, y0: torch.Tensor,
                params: Sequence[torch.Tensor]):
    c = tableau.c
    a = tableau.a
    b = tableau.b
    s = len(c)
    nt = len(t)
    dtype = t.dtype
    device = t.device

    # set up the results list
    yt_lst: List[torch.Tensor] = []
    yt_lst.append(y0)
    y = y0
    # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods
    # for the implementation
    for i in range(nt - 1):
        t0 = t[i]
        t1 = t[i + 1]
        h = t1 - t0
        ks: List[torch.Tensor] = []
        ksum: Union[float, torch.Tensor] = 0.0
        for j in range(s):
            if j == 0:
                k = fcn(t0, y, *params)
            else:
                ak: Union[float, torch.Tensor] = 0.0
                aj = a[j]
                for m in range(j):
                    ak = aj[m] * ks[m] + ak
                k = fcn(t0 + c[j] * h, h * ak + y, *params)
            ks.append(k)
            ksum = ksum + b[j] * k
        y = h * ksum + y
        yt_lst.append(y)
    yt = torch.stack(yt_lst, dim=0)
    return yt

############################# list of methods #############################
def rk38_ivp(fcn: Callable[..., torch.Tensor], t: torch.Tensor, y0: torch.Tensor,
             params: Sequence[torch.Tensor], **kwargs):
    return explicit_rk(rk38_tableau, fcn, t, y0, params)

def fwd_euler_ivp(fcn: Callable[..., torch.Tensor], t: torch.Tensor, y0: torch.Tensor,
                  params: Sequence[torch.Tensor], **kwargs):
    return explicit_rk(fwd_euler_tableau, fcn, t, y0, params)

def rk4_ivp(fcn: Callable[..., torch.Tensor], t: torch.Tensor, y0: torch.Tensor,
            params: Sequence[torch.Tensor], **kwargs):
    """
    Perform the Runge-Kutta steps of order 4 with a fixed step size.
    """
    return explicit_rk(rk4_tableau, fcn, t, y0, params)
