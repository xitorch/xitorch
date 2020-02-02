import torch
from lintorch.utils.misc import set_default_option

def solve(A, params, B, biases=None, fwd_options={}, bck_options={}):
    """
    Performing iterative method to solve the equation Ax=b or
    (A-biases*I)x=b.
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
    * fwd_options: dict
        Options of the iterative solver in the forward calculation
    * bck_options: dict
        Options of the iterative solver in the backward calculation
    """
    return solve_torchfcn.apply(A, B, biases, fwd_options, bck_options, *params)

class solve_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, biases, fwd_options, bck_options, *params):
        # B: (nbatch, nr, ncols)
        # biases: (nbatch, ncols) or None
        # params: (nbatch,...)
        # x: (nbatch, nc, ncols)
        config = set_default_option({
            "method": "conjgrad",
        }, fwd_options)
        ctx.bck_config = set_default_option({
            "method": "conjgrad",
        }, bck_options)

        # check symmetricity and must be a square matrix
        if not A.is_symmetric:
            msg = "The solve function cannot be used for non-symmetric transformation at the moment."
            raise RuntimeError(msg)
        if A.shape[0] != A.shape[1]:
            msg = "The solve function cannot be used for non-square transformation."
            raise RuntimeError(msg)

        method = config["method"].lower()
        if method == "conjgrad":
            x = conjgrad(A, params, B, biases=biases, **config)
        else:
            raise RuntimeError("Unknown solve method: %s" % config["method"])

        ctx.A = A
        ctx.biases = biases
        ctx.x = x
        ctx.params = params
        return x

    @staticmethod
    def backward(ctx, grad_x):
        # grad_x: (nbatch, nc, ncols)
        # ctx.x: (nbatch, nc, ncols)

        # detach x from the graph to avoid infinite loop with this function
        x = ctx.x.detach()

        # solve (A-biases*I)^T v = grad_x
        # this is the grad of B
        # (nbatch, nr, ncols)
        v = solve_torchfcn.apply(ctx.A, grad_x, ctx.biases, {}, ctx.bck_config,
            *ctx.params)
        grad_B = v

        # calculate the biases gradient
        grad_biases = None
        if ctx.biases is not None:
            grad_biases = (v * ctx.x).sum(dim=1) # (nbatch, ncols)

        # calculate the grad of matrices parameters
        with torch.enable_grad():
            loss = -ctx.A(x, *ctx.params) # (nbatch, nr, ncols)
        grad_params = torch.autograd.grad((loss,), ctx.params, grad_outputs=(v,),
            create_graph=torch.is_grad_enabled())

        return (None, grad_B, grad_biases, None, None, *grad_params)

def conjgrad(A, params, B, biases=None, **options):
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
        precond = lambda X: At.precond(X, *params, biases=biases)
    else:
        precond = lambda X: X

    # set up the biases
    if biases is not None:
        Aa = lambda X: At(X, *params) - X*biases.unsqueeze(1)
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
        alpha = Rs_old / _dot(P, Ap) # (nbatch, na, ncols)
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

        P = prR + (Rs_new / Rs_old) * P
        Rs_old = Rs_new

    return X

def _dot(C, D):
    return (C*D).sum(dim=1, keepdim=True) # (nbatch, 1, ncols)

if __name__ == "__main__":
    from lintorch.core.base import Module, module

    n = 1200
    dtype = torch.float64
    A1 = torch.rand(1,n,n).to(dtype) * 1e-2
    A2 = A1.transpose(-2,-1) + A1
    diag = torch.arange(n).to(dtype)+1.0 # (na,)
    Amat = A2 + diag.diag_embed()

    @module(shape=(n,n))
    def A(X):
        return torch.bmm(Amat, X)

    @A.set_precond
    def precond(X, biases=None):
        # X: (nbatch, na, ncols)
        return X / diag.unsqueeze(-1)

    xtrue = torch.rand(1,n,1).to(dtype)
    b = A(xtrue)
    xinv = solve(A, [], b,
        fwd_options={
            "verbose": True,
        })

    print((xinv - xtrue).abs().max())
    print(xinv - xtrue)
