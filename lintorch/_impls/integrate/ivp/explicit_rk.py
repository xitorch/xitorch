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

def explicit_rk(tableau, fcn, t, y0, params):
    c = tableau["c"]
    a = tableau["a"]
    b = tableau["b"]
    s = len(c)
    ny = len(y0)
    nt = len(t)
    dtype = t.dtype
    device = t.device

    # set up the results list
    yt = []
    for i in range(ny):
        y0i = y0[i]
        yti = torch.empty((nt, *y0i.shape), dtype=dtype, device=device)
        yti[0] = y0i
        yt.append(yti)

    y = y0
    # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods
    # for the implementation
    for i in range(nt-1):
        t0 = t[i]
        t1 = t[i+1]
        h = t1 - t0
        ks = []
        ksum = [0.0 for _ in range(ny)]
        for j in range(s):
            if j == 0:
                k = fcn(t0, y, *params)
            else:
                ak = [0.0 for _ in range(ny)]
                for m in range(j):
                    ak = _tuple_axpy1(a[j][m], ks[m], ak)
                k = fcn(t0 + c[j] * h, _tuple_axpy1(h, ak, y), *params)
            ks.append(k)
            ksum = _tuple_axpy1(b[j], k, ksum)
        y = _tuple_axpy1(h, ksum, y)
        for j in range(ny):
            yt[j][i+1] = y[j]
    return yt

############################# list of tableaus #############################
rk4_tableau = {
    "c": [0.0, 0.5, 0.5, 1.0],
    "b": [1/6., 1/3., 1/3., 1/6.],
    "a": [[0.0, 0.0, 0.0, 0.0],
          [0.5, 0.0, 0.0, 0.0],
          [0.0, 0.5, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0]]
}
rk38_tableau = {
    "c": [0.0, 1/3, 2/3, 1.0],
    "b": [1/8, 3/8, 3/8, 1/8],
    "a": [[ 0.0, 0.0, 0.0, 0.0],
          [ 1/3, 0.0, 0.0, 0.0],
          [-1/3, 1.0, 0.0, 0.0],
          [ 1.0,-1.0, 1.0, 0.0]]
}

############################# list of methods #############################
def rk38_ivp(fcn, t, y0, params, **kwargs):
    return explicit_rk(rk38_tableau, fcn, t, y0, params)

# explicit rk4 implementation to speed up
def rk4_ivp(fcn, t, y0, params, **kwargs):
    dtype = t.dtype
    device = t.device
    nt = torch.numel(t)
    ny = len(y0)

    # set up the results list
    yt = []
    for i in range(ny):
        y0i = y0[i]
        yti = torch.empty((nt, *y0i.shape), dtype=dtype, device=device)
        yti[0] = y0i
        yt.append(yti)

    y = y0
    for i in range(nt-1):
        t0 = t[i]
        t1 = t[i+1]
        h = t1 - t0
        h2 = h * 0.5
        k1 = fcn(t0, y, *params)
        k2 = fcn(t0 + h2, _tuple_axpy1(h2, k1, y), *params)
        k3 = fcn(t0 + h2, _tuple_axpy1(h2, k2, y), *params)
        k4 = fcn(t0 + h, _tuple_axpy1(h, k3, y), *params)
        ksum = _tuple_axpy1(2, k2, k1)
        ksum = _tuple_axpy1(2, k3, ksum)
        ksum = _tuple_axpy1(1, k4, ksum)
        y = _tuple_axpy1(h/6., ksum, y)
        for j in range(ny):
            yt[j][i+1] = y[j]
    return yt

def _tuple_axpy1(a, xs, ys): # a*x + y (only x and y are tuple)
    return [(a * x + y) for (x,y) in zip(xs, ys)]