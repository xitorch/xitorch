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

__all__ = ["rk4_ivp", "rk38_ivp"]

def explicit_rk(tableau, fcn, t, y0, params):
    c = tableau["c"]
    a = tableau["a"]
    b = tableau["b"]
    s = len(c)
    nt = len(t)
    dtype = t.dtype
    device = t.device

    # set up the results list
    yt = torch.empty((nt, *y0.shape), dtype=dtype, device=device)

    yt[0] = y0
    y = y0
    # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods
    # for the implementation
    for i in range(nt - 1):
        t0 = t[i]
        t1 = t[i + 1]
        h = t1 - t0
        ks = []
        ksum = 0.0
        for j in range(s):
            if j == 0:
                k = fcn(t0, y, *params)
            else:
                ak = 0.0
                for m in range(j):
                    ak = a[j][m] * ks[m] + ak
                k = fcn(t0 + c[j] * h, h * ak + y, *params)
            ks.append(k)
            ksum += b[j] * k
        y = h * ksum + y
        yt[i + 1] = y
    return yt


############################# list of tableaus #############################
rk4_tableau = {
    "c": [0.0, 0.5, 0.5, 1.0],
    "b": [1 / 6., 1 / 3., 1 / 3., 1 / 6.],
    "a": [[0.0, 0.0, 0.0, 0.0],
          [0.5, 0.0, 0.0, 0.0],
          [0.0, 0.5, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0]]
}
rk38_tableau = {
    "c": [0.0, 1 / 3, 2 / 3, 1.0],
    "b": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "a": [[0.0, 0.0, 0.0, 0.0],
          [1 / 3, 0.0, 0.0, 0.0],
          [-1 / 3, 1.0, 0.0, 0.0],
          [1.0, -1.0, 1.0, 0.0]]
}

############################# list of methods #############################
def rk38_ivp(fcn, t, y0, params, **kwargs):
    return explicit_rk(rk38_tableau, fcn, t, y0, params)

# explicit rk4 implementation to speed up
def rk4_ivp(fcn, t, y0, params, **kwargs):
    """
    Perform the Runge-Kutta steps of order 4 with a fixed step size.
    """
    dtype = t.dtype
    device = t.device
    nt = torch.numel(t)

    # set up the results
    yt = torch.empty((nt, *y0.shape), dtype=dtype, device=device)

    yt[0] = y0
    y = y0
    for i in range(nt - 1):
        t0 = t[i]
        t1 = t[i + 1]
        h = t1 - t0
        h2 = h * 0.5
        k1 = fcn(t0, y, *params)
        k2 = fcn(t0 + h2, h2 * k1 + y, *params)
        k3 = fcn(t0 + h2, h2 * k2 + y, *params)
        k4 = fcn(t0 + h, h  * k3 + y, *params)
        y = h / 6. * (k1 + 2 * k2 + 2 * k3 + k4) + y
        yt[i + 1] = y
    return yt
