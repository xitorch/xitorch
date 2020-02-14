import torch
from lintorch.utils.misc import set_default_option

__all__ = ["solve"]

def solve(A, params, B, biases=None, M=None, mparams=[],
          fwd_options={}, bck_options={}):
    """
    Performing iterative method to solve the equation Ax=b or
    (A-biases*M)x=b.
    This function can also solve batched multiple inverse equation at the
        same time by applying A to a tensor X with shape (nbatch, na, ncols).
    The applied biases are not necessarily identical for each column.

    Arguments
    ---------
    * A: lintorch.Module
        A function that takes an input X and produce the vectors in the same
        space as B. The matrix A must be symmetric.
    * params: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to A.forward(x,*params).
        Each of params must have shape of (nbatch,...)
    * B: torch.tensor (nbatch,na,ncols)
        The tensor on the right hand side.
    * biases: torch.tensor (nbatch,ncols) or None
        If not None, it will solve (A-biases*I)*X = B. Otherwise, it just solves
        A*X = B. biases would be applied to every column.
    * M: lintorch.Module or None
        The transformation on the biases side. If biases is None,
        then this argument is ignored. If None or ignored, then M=I.
    * mparams: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to M.
    * fwd_options: dict
        Options of the iterative solver in the forward calculation
    * bck_options: dict
        Options of the iterative solver in the backward calculation
    """
    na = len(params)
    if biases is None:
        M = None
    return solve_torchfcn.apply(A, B, biases, M, fwd_options, bck_options, na, *params, *mparams)

class solve_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, biases, M, fwd_options, bck_options, na, *AMparams):
        # B: (nbatch, nr, ncols)
        # biases: (nbatch, ncols) or None
        # params: (nbatch,...)
        # x: (nbatch, nc, ncols)

        # separate the parameters for A and for M
        params = AMparams[:na]
        mparams = AMparams[na:]

        config = set_default_option({
            "method": "conjgrad",
        }, fwd_options)
        ctx.bck_config = set_default_option({
            "method": "conjgrad",
        }, bck_options)

        # check the shape of the matrix
        if A.shape[0] != A.shape[1]:
            msg = "The solve function cannot be used for non-square transformation."
            raise RuntimeError(msg)

        method = config["method"].lower()
        if method == "conjgrad":
            x = conjgrad(A, params, B, biases=biases, M=M, mparams=mparams, **config)
        elif method == "lbfgs":
            x = lbfgs(A, params, B, biases=biases, M=M, mparams=mparams, **config)
        else:
            raise RuntimeError("Unknown solve method: %s" % config["method"])

        ctx.A = A
        ctx.M = M
        ctx.biases = biases
        ctx.x = x
        ctx.params = params
        ctx.mparams = mparams
        return x

    @staticmethod
    def backward(ctx, grad_x):
        # grad_x: (nbatch, nc, ncols)
        # ctx.x: (nbatch, nc, ncols)

        # solve (A-biases*M)^T v = grad_x
        # this is the grad of B
        # (nbatch, nr, ncols)
        v = solve(ctx.A, ctx.params, grad_x,
            biases=ctx.biases, M=ctx.M, mparams=ctx.mparams,
            fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
        grad_B = v

        # calculate the biases gradient
        grad_biases = None
        if ctx.biases is not None:
            if ctx.M is None:
                Mx = ctx.x
            else:
                Mx = ctx.M(ctx.x, *ctx.mparams)
            grad_biases = (v * Mx).sum(dim=1) # (nbatch, ncols)

        # calculate the grad of matrices parameters
        params = [p.clone().detach().requires_grad_() for p in ctx.params]
        with torch.enable_grad():
            loss = -ctx.A(ctx.x, *params) # (nbatch, nr, ncols)

        grad_params = torch.autograd.grad((loss,), params, grad_outputs=(v,),
            create_graph=torch.is_grad_enabled())

        # calculate the gradient to the biases matrices
        grad_mparams = []
        if ctx.M is not None:
            mparams = [p.clone().detach().requires_grad_() for p in ctx.mparams]
            with torch.enable_grad():
                lmbdax = ctx.x * ctx.biases.unsqueeze(1)
                mloss = ctx.M(lmbdax, *mparams)

            grad_mparams = torch.autograd.grad((mloss,), mparams,
                grad_outputs=(v,),
                create_graph=torch.is_grad_enabled())

        return (None, grad_B, grad_biases, None, None, None, None,
                *grad_params, *grad_mparams)

def conjgrad(A, params, B, biases=None, M=None, mparams=[], **options):
    # use conjugate gradient descent to solve the inverse equation
    nbatch, na, ncols = B.shape
    config = set_default_option({
        "max_niter": na,
        "verbose": False,
        "min_eps": 1e-6, # minimum residual to stop
    }, options)

    # this function cannot work for non-symmetric matrix
    if not A.is_symmetric:
        raise RuntimeError("This function only works for real-symmetric matrix.")

    # set up the preconditioning
    At = A
    if At.is_precond_set():
        precond = lambda X: At.precond(X, *params, biases=biases,
                                       M=M, mparams=mparams)
    else:
        precond = lambda X: X

    # set up the biases
    if biases is not None:
        b = biases.unsqueeze(1)
        if M is not None:
            Aa = lambda X: At(X, *params) - M(X, *mparams) * b
        else:
            Aa = lambda X: At(X, *params) - X * b
    else:
        Aa = lambda X: At(X, *params)

    # double the transformation to ensure posdefness
    precondt = precond
    B = Aa(B)
    A = lambda X: Aa(Aa(X))
    precond = lambda X: precondt(precondt(X))

    # assign a variable to some of the options
    verbose = config["verbose"]
    min_eps = config["min_eps"]

    # initialize the guess
    X = torch.zeros_like(B).to(B.device)
    if torch.allclose(B, X):
        return X

    # do the iterations
    R = B - A(X)
    P = precond(R) # (nbatch, na, ncols)
    Rs_old = _dot(R, P) # (nbatch, 1, ncols)
    for i in range(config["max_niter"]):
        Ap = A(P) # (nbatch, na, ncols)
        alpha = _safe_divide(Rs_old, _dot(P, Ap)) # (nbatch, na, ncols)
        X = X + alpha * P
        R = R - alpha * Ap
        prR = precond(R)
        Rs_new = _dot(R, prR)

        # check convergence
        eps_max = Rs_new.abs().max()
        if verbose and (i+1)%1 == 0:
            print("Iter %d: %.3e" % (i+1, eps_max))
        if eps_max < min_eps:
            break

        P = prR + _safe_divide(Rs_new, Rs_old) * P
        Rs_old = Rs_new

    return X

def lbfgs(A, params, B, biases=None, M=None, mparams=[], **options):
    # B: (nbatch, na, ncols)
    # biases: (nbatch, ncols)
    nbatch, na, ncols = B.shape

    config = set_default_option({
        "max_niter": 20,
        "min_eps": 1e-6,
        "max_memory": 10,
        "alpha0": 1.0,
        "jinv0": 1.0,
        "linesearch": False,
        "verbose": False,
    }, options)

    def f(x):
        # x: (nbatch * ncols, na)
        # biases: (nbatch, ncols)
        x = x.view(nbatch, ncols, na).transpose(-2,-1) # (nbatch, na, ncols)
        y = A(x, *params) # (nbatch, na, ncols)
        if biases is not None:
            b = x
            if M is not None:
                b = M(b, *mparams) # (nbatch, na, ncols)
            b = b * biases.unsqueeze(1)
            y = y - b # (nbatch, na, ncols)
        res = (y - B).transpose(-2,-1).contiguous().view(-1, na)
        return res

    # pull out the options for fast access
    min_eps = config["min_eps"]
    max_memory = config["max_memory"]
    verbose = config["verbose"]
    linesearch = config["linesearch"]
    alpha = config["alpha0"]

    # set up the initial jinv and the memories
    b = B.transpose(-2,-1).view(nbatch*na, ncols)
    x0 = torch.zeros_like(b).to(b.device)
    H0 = _set_jinv0_diag(jinv0, b) # (nbatch*ncols, na)
    sk_history = []
    yk_history = []
    rk_history = []

    def _apply_Vk(rk, sk, yk, grad):
        # sk: (nbatch, nfeat)
        # yk: (nbatch, nfeat)
        # rk: (nbatch, 1)
        return grad - (sk * grad).sum(dim=-1, keepdim=True) * rk * yk

    def _apply_VkT(rk, sk, yk, grad):
        # sk: (nbatch, nfeat)
        # yk: (nbatch, nfeat)
        # rk: (nbatch, 1)
        return grad - (yk * grad).sum(dim=-1, keepdim=True) * rk * sk

    def _apply_Hk(H0, sk_hist, yk_hist, rk_hist, gk):
        # H0: (nbatch, nfeat)
        # sk: (nbatch, nfeat)
        # yk: (nbatch, nfeat)
        # rk: (nbatch, 1)
        # gk: (nbatch, nfeat)
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
    for k in range(config["max_niter"]):
        dk = -_apply_Hk(H0, sk_history, yk_history, rk_history, gk)
        xknew, gknew = _line_search(xk, gk, dk, f)

        # store the history
        sk = xknew - xk # (nbatch, nfeat)
        yk = gknew - gk
        inv_rhok = 1.0 / (sk * yk).sum(dim=-1, keepdim=True) # (nbatch, 1)
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

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (k+1, gk.abs().max()))
        if torch.allclose(gk, torch.zeros_like(gk), atol=min_feps):
            break

    # xk: (nbatch*ncols, na)
    res = xk.view(nbatch, ncols, na).transpose(-2,-1)
    return res

def _set_jinv0_diag(jinv0, x0):
    if type(jinv0) == torch.Tensor:
        jinv = jinv0
    else:
        jinv = torch.zeros_like(x0).to(x0.device) + jinv0
    return jinv

def _safe_divide(A, B, eps=1e-10):
    C = B.clone()
    C[C.abs() < eps] = eps
    return A / C

def _dot(C, D):
    return (C*D).sum(dim=1, keepdim=True) # (nbatch, 1, ncols)

if __name__ == "__main__":
    import time
    from lintorch.core.base import Module, module
    from lintorch.utils.fd import finite_differences

    n = 20
    dtype = torch.float64
    torch.manual_seed(123)
    A1 = (torch.rand(1,n,n).to(dtype) * 1e-2).requires_grad_()
    diag = (torch.arange(n).to(dtype)+1.0).requires_grad_() # (na,)

    @module(shape=(n,n))
    def A(X, A1, diag):
        Amat = A1.transpose(-2,-1) + A1 + diag.diag_embed()
        return torch.bmm(Amat, X)

    @A.set_precond
    def precond(X, A1, diag, biases=None, M=None, mparams=[]):
        # X: (nbatch, na, ncols)
        return X / diag.unsqueeze(-1)

    xtrue = torch.rand(1,n,1).to(dtype)
    A = A.to(dtype)
    b = A(xtrue, A1, diag).detach().requires_grad_()
    biases = (torch.ones((b.shape[0], b.shape[-1]))*1.2).to(dtype).requires_grad_()
    def getloss(A1, diag, b, biases):
        fwd_options = {
            "verbose": False,
            "min_eps": 1e-9
        }
        bck_options = {
            "verbose": False,
        }
        with torch.enable_grad():
            A1.requires_grad_()
            b.requires_grad_()
            diag.requires_grad_()
            biases.requires_grad_()
            xinv = solve(A, (A1, diag), b, biases=biases, fwd_options=fwd_options)
            lss = (xinv**2).sum()
            grad_A1, grad_diag, grad_b, grad_biases = torch.autograd.grad(
                lss,
                (A1, diag, b, biases), create_graph=True)
        loss = 0
        loss = loss + (grad_A1**2).sum()
        loss = loss + (grad_diag**2).sum()
        # loss = loss + (grad_b**2).sum()
        # loss = loss + (grad_biases**2).sum()
        return loss

    t0 = time.time()
    loss = getloss(A1, diag, b, biases)
    t1 = time.time()
    print("Forward done in %fs" % (t1 - t0))
    loss.backward()
    t2 = time.time()
    print("Backward done in %fs" % (t2 - t1))
    Agrad = A1.grad.data
    dgrad = diag.grad.data
    bgrad = b.grad.data
    biasesgrad = biases.grad.data

    Afd = finite_differences(getloss, (A1, diag, b, biases), 0, eps=1e-4)
    dfd = finite_differences(getloss, (A1, diag, b, biases), 1, eps=1e-5)
    bfd = finite_differences(getloss, (A1, diag, b, biases), 2, eps=1e-5)
    biasesfd = finite_differences(getloss, (A1, diag, b, biases), 3, eps=1e-5)
    print("Finite differences done")

    print("A1:")
    print(Agrad)
    print(Afd)
    print(Agrad/Afd)

    print("diag:")
    print(dgrad)
    print(dfd)
    print(dgrad/dfd)

    print("B:")
    print(bgrad)
    print(bfd)
    print(bgrad/bfd)

    print("biases:")
    print(biasesgrad)
    print(biasesfd)
    print(biasesgrad/biasesfd)
