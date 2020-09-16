import torch
import numpy as np
import warnings
from xitorch._utils.misc import set_default_option
from xitorch._impls.optimize.linesearch import line_search

def selfconsistent(f, x0, jinv0=1.0, **options):
    """
    Solve the root finder problem with Broyden's method.

    Arguments
    ---------
    * f: callable
        Callable that takes params as the input and output nfeat-outputs.
    * x0: torch.tensor (nbatch, nfeat)
        Initial value of parameters to be put in the function, f.
    * jinv0: float or torch.tensor (nbatch, nfeat, nfeat)
        The initial inverse of the Jacobian. If float, it will be the diagonal.
    * options: dict or None
        Options of the function.

    Returns
    -------
    * x: torch.tensor (nbatch, nfeat)
        The x that approximate f(x) = 0.
    """
    # set up the default options
    config = set_default_option({
        "max_niter": 20,
        "min_eps": 1e-6,
        "beta": 0.9, # contribution of the new delta_n to the total delta_n
        "jinvdecay": 1.0,
        "decayevery": 100,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_eps = config["min_eps"]
    verbose = config["verbose"]
    beta = config["beta"]
    jinvdecay = config["jinvdecay"]
    decayevery = config["decayevery"]

    # pull out the parameters of x0
    nbatch, nfeat = x0.shape
    device = x0.device
    dtype = x0.dtype

    # set up the initial jinv
    jinv = _set_jinv0(jinv0, x0)

    # perform the Broyden iterations
    x = x0
    fx = f(x0) # (nbatch, nfeat)
    stop_reason = "max_niter"
    dx = torch.zeros_like(x).to(x.device)
    bestcrit = float("inf")
    for i in range(config["max_niter"]):
        dxnew = -jinv0 * fx # (nbatch, nfeat)
        dx = (1 - beta) * dx + beta * dxnew
        xnew = x + dx # (nbatch, nfeat)
        fxnew = f(xnew)
        dfnew = fxnew - fx

        # update variables for the next iteration
        fx = fxnew
        x = xnew
        if (i+1) % decayevery == 0:
            jinv = jinv * jinvdecay

        # get the best results
        crit = fx.abs().max()
        if crit < bestcrit:
            bestcrit = crit
            bestx = x

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (i+1, crit))
        if torch.allclose(fx, torch.zeros_like(fx), atol=min_eps):
            stop_reason = "min_eps"
            break

    if stop_reason != "min_eps":
        msg = "The selfconsistent iteration does not converge to the required accuracy."
        msg += "\nRequired: %.3e. Achieved: %.3e" % (min_eps, bestcrit)
        warnings.warn(msg)

    return bestx

def diis(f, x0, jinv0=1.0, **options):
    """
    Solve the root finder problem with DIIS method.

    Arguments
    ---------
    * f: callable
        Callable that takes params as the input and output nfeat-outputs.
    * x0: torch.tensor (nbatch, nfeat)
        Initial value of parameters to be put in the function, f.
    * jinv0: float or torch.tensor (nbatch, nfeat, nfeat)
        The initial inverse of the Jacobian. If float, it will be the diagonal.
    * options: dict or None
        Options of the function.

    Returns
    -------
    * x: torch.tensor (nbatch, nfeat)
        The x that approximate f(x) = 0.
    """
    # set up the default options
    config = set_default_option({
        "max_niter": 20,
        "min_eps": 1e-6,
        "max_memory": 20,
        "minit": 10,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_eps = config["min_eps"]
    verbose = config["verbose"]
    max_memory = config["max_memory"]
    minit = config["minit"]

    # pull out the parameters of x0
    nbatch, nfeat = x0.shape
    device = x0.device
    dtype = x0.dtype

    # set up the initial jinv
    jinv = _set_jinv0(jinv0, x0)

    # perform the iterations
    x = x0
    fx = f(x0) # (nbatch, nfeat)
    stop_reason = "max_niter"
    bestcrit = float("inf")
    nbatch, nfeat = fx.shape
    x_history = torch.empty((nbatch, max_memory, nfeat), dtype=x.dtype, device=x.device)
    e_history = torch.empty((nbatch, max_memory, nfeat), dtype=x.dtype, device=x.device)
    mfill = 0
    midx = 0
    for i in range(config["max_niter"]):
        if mfill < 2:
            dxnew = -jinv * fx # (nbatch, nfeat)
            xnew = x + dxnew # (nbatch, nfeat)
        else:
            # construct the matrix B
            fx_tensor = e_history[:,:mfill,:] # (nbatch, m, nfeat)
            Bul = torch.matmul(fx_tensor, fx_tensor.transpose(-2,-1)) # (nbatch, m, m)
            Bu = torch.cat((Bul, -torch.ones(Bul.shape[0],Bul.shape[1],1, dtype=Bul.dtype, device=Bul.device)), dim=-1) # (nbatch, m, m+1)
            B = torch.cat((Bu, -torch.ones(Bu.shape[0],1,Bu.shape[-1], dtype=Bu.dtype, device=Bu.device)), dim=-2) # (nbatch, m+1, m+1)
            B[:,-1,-1] = 0.0

            # solve the linear equation to get the coefficients
            a = torch.zeros(B.shape[0], B.shape[1], 1, dtype=B.dtype, device=B.device) # (nbatch, m+1, 1)
            a[:,-1] = -1.0
            c = torch.solve(a, B)[0].squeeze(-1)[:,:mfill] # (nbatch, m)

            x_tensor = x_history[:,:mfill,:] # (nbatch, m, nfeat)
            xnew = (x_tensor * c.unsqueeze(-1)).sum(dim=1) # (nbatch, m)

        fxnew = f(xnew) # (nbatch, nfeat)
        dx = xnew - x

        # update variables for the next iteration
        fx = fxnew
        x = xnew

        # add the history
        if i >= minit:
            x_history[:,midx,:] = x
            # e_history[:,midx,:] = dx
            e_history[:,midx,:] = fx
            midx = (midx + 1) % max_memory
            mfill = (mfill + 1) if mfill < max_memory else mfill

        # get the best results
        crit = fx.abs().max()
        if crit < bestcrit:
            bestcrit = crit
            bestx = x

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (i+1, crit))
        if torch.allclose(fx, torch.zeros_like(fx), atol=min_eps):
            stop_reason = "min_eps"
            break

    if stop_reason != "min_eps":
        msg = "The DIIS iteration does not converge to the required accuracy."
        msg += "\nRequired: %.3e. Achieved: %.3e" % (min_eps, bestcrit)
        warnings.warn(msg)

    return bestx

def broyden(f, x0, jinv0=1.0, **options):
    """
    Solve the root finder problem with Broyden's method.

    Arguments
    ---------
    * f: callable
        Callable that takes params as the input and output nfeat-outputs.
    * x0: torch.tensor (nbatch, nfeat)
        Initial value of parameters to be put in the function, f.
    * jinv0: float or torch.tensor (nbatch, nfeat, nfeat)
        The initial inverse of the Jacobian. If float, it will be the diagonal.
    * options: dict or None
        Options of the function.

    Returns
    -------
    * x: torch.tensor (nbatch, nfeat)
        The x that approximate f(x) = 0.
    """
    raise RuntimeError("This method is unfinished. Please use other methods.")

    # set up the default options
    config = set_default_option({
        "max_niter": 20,
        "min_eps": 1e-6,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_eps = config["min_eps"]
    verbose = config["verbose"]

    # pull out the parameters of x0
    nbatch, nfeat = x0.shape
    device = x0.device
    dtype = x0.dtype

    # set up the initial jinv
    jinv = _set_jinv0(jinv0, x0)

    # perform the Broyden iterations
    x = x0
    fx = f(x0) # (nbatch, nfeat)
    stop_reason = "max_niter"
    for i in range(config["max_niter"]):
        dxnew = -jinv * fx # (nbatch, nfeat)
        xnew = x + dxnew # (nbatch, nfeat)
        fxnew = f(xnew)
        dfnew = fxnew - fx

        # calculate the new jinv
        xtnew_jinv = torch.bmm(xnew.unsqueeze(1), jinv) # (nbatch, 1, nfeat)
        jinv_dfnew = torch.bmm(jinv, dfnew.unsqueeze(-1)) # (nbatch, nfeat, 1)
        xtnew_jinv_dfnew = torch.bmm(xtnew_jinv, dfnew.unsqueeze(-1)) # (nbatch, 1, 1)
        jinvnew = jinv + torch.bmm(dxnew - jinv_dfnew, xtnew_jinv) / xtnew_jinv_dfnew

        # update variables for the next iteration
        fx = fxnew
        jinv = jinvnew
        x = xnew

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (i+1, fx.abs().max()))
        if torch.allclose(fx, torch.zeros_like(fx), atol=min_eps):
            stop_reason = "min_eps"
            break

    if stop_reason != "min_eps":
        msg = "The Broyden iteration does not converge to the required accuracy."
        msg += "\nRequired: %.3e. Achieved: %.3e" % (min_eps, fx.abs().max())
        warnings.warn(msg)

    return x

def lbfgs(f, x0, jinv0=1.0, **options):
    """
    Solve the root finder problem with L-BFGS method.

    Arguments
    ---------
    * f: callable
        Callable that takes params as the input and output nfeat-outputs.
    * x0: torch.tensor (*, nfeat)
        Initial value of parameters to be put in the function, f.
    * jinv0: float or torch.tensor (nbatch, nfeat, nfeat)
        The initial inverse of the Jacobian. If float, it will be the diagonal.
    * options: dict or None
        Options of the function.

    Returns
    -------
    * x: torch.tensor (nbatch, nfeat)
        The x that approximate f(x) = 0.
    """
    config = set_default_option({
        "max_niter": 20,
        "min_eps": 1e-6,
        "max_memory": 10,
        "alpha0": 1.0,
        "linesearch": False,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_eps = config["min_eps"]
    max_memory = config["max_memory"]
    verbose = config["verbose"]
    linesearch = config["linesearch"]
    alpha = config["alpha0"]

    # set up the initial jinv and the memories
    H0 = _set_jinv0_diag(jinv0, x0) # (*, nfeat)
    sk_history = []
    yk_history = []
    rk_history = []

    def _apply_Vk(rk, sk, yk, grad):
        # sk: (*, nfeat)
        # yk: (*, nfeat)
        # rk: (*, 1)
        return grad - (sk * grad).sum(dim=-1, keepdim=True) * rk * yk

    def _apply_VkT(rk, sk, yk, grad):
        # sk: (*, nfeat)
        # yk: (*, nfeat)
        # rk: (*, 1)
        return grad - (yk * grad).sum(dim=-1, keepdim=True) * rk * sk

    def _apply_Hk(H0, sk_hist, yk_hist, rk_hist, gk):
        # H0: (*, nfeat)
        # sk: (*, nfeat)
        # yk: (*, nfeat)
        # rk: (*, 1)
        # gk: (*, nfeat)
        nhist = len(sk_hist)
        if nhist == 0:
            return H0 * gk

        k = nhist - 1
        rk = rk_hist[k]
        sk = sk_hist[k]
        yk = yk_hist[k]

        # get the last term (rk * sk * sk.T)
        rksksk = (sk * gk).sum(dim=-1, keepdim=True) * rk * sk

        # calculate the V_(k-1)
        grad = gk
        grad = _apply_Vk(rk_hist[k], sk_hist[k], yk_hist[k], grad)
        grad = _apply_Hk(H0, sk_hist[:k], yk_hist[:k], rk_hist[:k], grad)
        grad = _apply_VkT(rk_hist[k], sk_hist[k], yk_hist[k], grad)
        return grad + rksksk

    def _line_search(xk, gk, dk, g):
        if linesearch:
            dx, dg, nit = line_search(dk, xk, gk, g)
            return xk + dx, gk + dg
        else:
            return xk + alpha*dk, g(xk + alpha*dk)

    # perform the main iteration
    xk = x0
    gk = f(xk)
    bestgk = gk.abs().max()
    bestx = x0
    stop_reason = "max_niter"
    for k in range(config["max_niter"]):
        dk = -_apply_Hk(H0, sk_history, yk_history, rk_history, gk)
        xknew, gknew = _line_search(xk, gk, dk, f)

        # store the history
        sk = xknew - xk # (*, nfeat)
        yk = gknew - gk
        inv_rhok = 1.0 / (sk * yk).sum(dim=-1, keepdim=True) # (*, 1)
        sk_history.append(sk)
        yk_history.append(yk)
        rk_history.append(inv_rhok)
        if len(sk_history) > max_memory:
            sk_history = sk_history[-max_memory:]
            yk_history = yk_history[-max_memory:]
            rk_history = rk_history[-max_memory:]

        # update for the next iteration
        xk = xknew
        # alphakold = alphak
        gk = gknew

        # save the best point
        maxgk = gk.abs().max()
        if maxgk < bestgk:
            bestx = xk
            bestgk = maxgk

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (k+1, gk.abs().max()))
        if maxgk < min_eps:
            stop_reason = "min_eps"
            break

    if stop_reason != "min_eps":
        msg = "The L-BFGS iteration does not converge to the required accuracy."
        msg += "\nRequired: %.3e. Achieved: %.3e" % (min_eps, bestgk)
        warnings.warn(msg)

    return bestx

def gradrca(f, x0, jinv0=1.0, **options):
    # set up the default options
    config = set_default_option({
        "max_niter": 20,
        "norders": 2,
        "min_eps": 1e-6,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_eps = config["min_eps"]
    verbose = config["verbose"]
    norders = config["norders"]

    # pull out the parameters of x0
    nbatch, nfeat = x0.shape
    device = x0.device
    dtype = x0.dtype

    # set up the initial jinv
    jinv = _set_jinv0(jinv0, x0)

    x = x0
    onesvec = torch.ones_like(x0).unsqueeze(-1).to(x0.device) / np.sqrt(nfeat) # (nbatch, nfeat, 1)
    for i in range(config["max_niter"]):
        xg = x.detach().requires_grad_()
        with torch.enable_grad():
            dx = f(xg)
            vunit = (dx / dx.norm(dim=-1, keepdim=True)).detach() # (nbatch, nfeat)
            loss = (dx * dx).sum(dim=-1)
            derivs = [loss]
            for j in range(norders):
                dldx = torch.autograd.grad(derivs[-1].sum(), (xg,), create_graph=(j<norders-1))[0]
                dldlmbda = (dldx * vunit).sum(dim=-1)
                derivs.append(dldlmbda)

        if norders == 2:
            dstep = (-derivs[1] / derivs[2]) # (nbatch,)
        elif norders == 3:
            dstep = (-derivs[2] + torch.sqrt(derivs[2]*derivs[2] - 2*derivs[1]*derivs[3])) / (derivs[3])
        else:
            raise RuntimeError("Order 4 or higher is not defined.")
        dstep = (jinv * dstep.unsqueeze(-1) * vunit)

        if verbose:
            print("Iter %d: %.3e" % (i+1, dx.detach().abs().max()))

        x = x + dstep
    return x

def _set_jinv0(jinv0, x0):
    nbatch, nfeat = x0.shape
    dtype = x0.dtype
    device = x0.device
    if type(jinv0) == torch.Tensor:
        jinv = jinv0
    else:
        jinv = torch.zeros(1,nfeat).to(dtype).to(device) + jinv0 # (1, nfeat)
    return jinv

def _set_jinv0_diag(jinv0, x0):
    if type(jinv0) == torch.Tensor:
        jinv = jinv0
    else:
        jinv = torch.zeros_like(x0).to(x0.device) + jinv0
    return jinv

if __name__ == "__main__":

    dtype = torch.float
    A = torch.tensor([[[0.9703, 0.1178, 0.5345],
         [0.0629, 0.3352, 0.6431],
         [0.8756, 0.7564, 0.1121]]]).to(dtype)
    xtrue = torch.tensor([[0.8690, 0.4324, 0.9035]]).to(dtype)
    b = torch.bmm(A, xtrue.unsqueeze(-1)).squeeze(-1)

    def f(x):
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b

    x0 = torch.zeros_like(xtrue)
    jinv0 = 1.0
    x = lbfgs(f, x0, jinv0, verbose=True)
    # x = broyden(f, x0, jinv0, verbose=True)

    print(A)
    print(x)
    print(xtrue)
    print(f(x))
