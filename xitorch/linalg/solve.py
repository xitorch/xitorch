import torch
import warnings
from typing import Union, Any, Mapping
import numpy as np
from scipy.sparse.linalg import gmres
from lintorch.linalg.linop import LinearOperator
from lintorch._impls.optimize.rootfinder import lbfgs, broyden
from lintorch._utils.bcast import normalize_bcast_dims, get_bcasted_dims
from lintorch._utils.assertfuncs import assert_runtime
from lintorch._utils.misc import set_default_option, dummy_context_manager
from lintorch.debug.modes import is_debug_enabled

def solve(A:LinearOperator, B:torch.Tensor, E:Union[torch.Tensor,None]=None,
          M:Union[LinearOperator,None]=None, posdef=False,
          fwd_options:Mapping[str,Any]={},
          bck_options:Mapping[str,Any]={}):
    """
    Performing iterative method to solve the equation AX=B or
    AX-MXE=B, where E is a diagonal matrix.
    This function can also solve batched multiple inverse equation at the
    same time by applying A to a tensor X with shape (...,na,ncols).
    The applied E are not necessarily identical for each column.

    Arguments
    ---------
    * A: lintorch.LinearOperator instance with shape (*BA, na, na)
        A function that takes an input X and produce the vectors in the same
        space as B.
    * B: torch.tensor (*BB, na, ncols)
        The tensor on the right hand side.
    * E: torch.tensor (*BE, ncols) or None
        If not None, it will solve AX-MXE = B. Otherwise, it just solves
        AX = B and M is ignored. E would be applied to every column.
    * M: lintorch.LinearOperator instance (*BM, na, na) or None
        The transformation on the E side. If E is None,
        then this argument is ignored. I E is not None and M is None, then M=I.
        This LinearOperator must be Hermitian.
    * fwd_options: dict
        Options of the iterative solver in the forward calculation
    * bck_options: dict
        Options of the iterative solver in the backward calculation
    """
    assert_runtime(A.shape[-1] == A.shape[-2], "The linear operator A must have a square shape")
    assert_runtime(A.shape[-1] == B.shape[-2], "Mismatch shape of A & B (A: %s, B: %s)" % (A.shape, B.shape))
    if M is not None:
        assert_runtime(M.shape[-1] == M.shape[-2], "The linear operator M must have a square shape")
        assert_runtime(M.shape[-1] == A.shape[-1], "The shape of A & M must match (A: %s, M: %s)" % (A.shape, M.shape))
        assert_runtime(M.is_hermitian, "The linear operator M must be a Hermitian matrix")
    if E is not None:
        assert_runtime(E.shape[-1] == B.shape[-1], "The last dimension of E & B must match (E: %s, B: %s)" % (E.shape, B.shape))
    if E is None and M is not None:
        warnings.warn("M is supplied but will be ignored because E is not supplied")

    # perform expensive check if debug mode is enabled
    if is_debug_enabled():
        A.check()
        if M is not None:
            M.check()

    if "method" not in fwd_options or fwd_options["method"].lower() == "exactsolve":
        return exactsolve(A, B, E, M)
    else:
        # get the unique parameters of A
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return solve_torchfcn.apply(
            A, B, E, M, posdef,
            fwd_options, bck_options,
            na, *params, *mparams)

class solve_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, E, M, posdef,
                fwd_options, bck_options,
                na, *all_params):
        # A: (*BA, nr, nr)
        # B: (*BB, nr, ncols)
        # E: (*BE, ncols) or None
        # M: (*BM, nr, nr) or None
        # all_params: list of tensor of any shape
        # returns: (*BABEM, nr, ncols)

        # separate the parameters for A and for M
        params = all_params[:na]
        mparams = all_params[na:]

        config = set_default_option({
        }, fwd_options)
        ctx.bck_config = set_default_option({
        }, bck_options)

        method = config["method"].lower()

        if torch.all(B == 0): # special case
            dims = (*_get_batchdims(A, B, E, M), *B.shape[-2:])
            x = torch.zeros(dims, dtype=B.dtype, device=B.device)
        elif method == "custom_exactsolve":
            x = custom_exactsolve(A, params, B, E=E, M=M, mparams=mparams, **config)
        elif method == "gmres":
            x = wrap_gmres(A, params, B, E=E, M=M, mparams=mparams, posdef=posdef, **config)
        elif method in ["lbfgs", "broyden"]:
            x = rootfinder_solve(method, A, params, B, E=E, M=M, mparams=mparams, posdef=posdef, **config)
        else:
            raise RuntimeError("Unknown solve method: %s" % config["method"])

        ctx.A = A
        ctx.M = M
        ctx.E = E
        ctx.x = x
        ctx.posdef = posdef
        ctx.params = params
        ctx.mparams = mparams
        ctx.na = na
        return x

    @staticmethod
    def backward(ctx, grad_x):
        # grad_x: (*BABEM, nr, ncols)
        # ctx.x: (*BABEM, nr, ncols)

        # solve (A-biases*M)^T v = grad_x
        # this is the grad of B
        AT = ctx.A.H # (*BA, nr, nr)
        MT = ctx.M.H if ctx.M is not None else None # (*BM, nr, nr)
        with AT.uselinopparams(*ctx.params), MT.uselinopparams(*ctx.mparams) if MT is not None else dummy_context_manager():
            v = solve(AT, grad_x, ctx.E, MT, posdef=ctx.posdef,
                fwd_options=ctx.bck_config, bck_options=ctx.bck_config) # (*BABEM, nr, ncols)
        grad_B = v

        # calculate the grad of matrices parameters
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in ctx.params]
            with ctx.A.uselinopparams(*params):
                loss = -ctx.A.mm(ctx.x) # (*BABEM, nr, ncols)

        grad_params = torch.autograd.grad((loss,), params, grad_outputs=(v,),
            create_graph=torch.is_grad_enabled())

        # calculate the biases gradient
        grad_E = None
        if ctx.E is not None:
            if ctx.M is None:
                Mx = ctx.x
            else:
                with ctx.M.uselinopparams(*ctx.mparams):
                    Mx = ctx.M.mm(ctx.x) # (*BABEM, nr, ncols)
            grad_E = torch.einsum('...rc,...rc->...c', v, Mx) # (*BABEM, ncols)

        # calculate the gradient to the biases matrices
        grad_mparams = []
        if ctx.M is not None and ctx.E is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in ctx.mparams]
                lmbdax = ctx.x * ctx.E.unsqueeze(-2)
                with ctx.M.uselinopparams(*mparams):
                    mloss = ctx.M.mm(lmbdax)

            grad_mparams = torch.autograd.grad((mloss,), mparams,
                grad_outputs=(v,),
                create_graph=torch.is_grad_enabled())

        return (None, grad_B, grad_E, None, None, None, None, None,
                *grad_params, *grad_mparams)

def wrap_gmres(A, params, B, E=None, M=None, mparams=[], posdef=False, **options):
    # A: (*BA, nr, nr)
    # B: (*BB, nr, ncols)
    # E: (*BE, ncols) or None
    # M: (*BM, nr, nr) or None

    # NOTE: currently only works for batched B (1 batch dim), but unbatched A
    assert len(A.shape) == 2 and len(B.shape) == 3, "Currently only works for batched B (1 batch dim), but unbatched A"

    # check the parameters
    msg = "GMRES can only do AX=B"
    assert A.shape[-2] == A.shape[-1], "GMRES can only work for square operator for now"
    assert E is None, msg
    assert M is None, msg

    # set the default config options
    nbatch, na, ncols = B.shape
    config = set_default_option({
        "min_eps": 1e-9,
        "max_niter": 2*na,
    }, options)
    min_eps = config["min_eps"]
    max_niter = config["max_niter"]

    B = B.transpose(-1,-2) # (nbatch, ncols, na)

    # convert the numpy/scipy
    with A.uselinopparams(*params):
        op = A.scipy_linalg_op()
        B_np = B.detach().numpy()
        res_np = np.empty(B.shape, dtype=np.float64)
        for i in range(nbatch):
            for j in range(ncols):
                x, info = gmres(op, B_np[i,j,:], tol=min_eps, atol=1e-12, maxiter=max_niter)
                if info > 0:
                    msg = "The GMRES iteration does not converge to the desired value "\
                          "(%.3e) after %d iterations" % \
                          (config["min_eps"], info)
                    warnings.warn(msg)
                res_np[i,j,:] = x

        res = torch.tensor(res_np, dtype=B.dtype, device=B.device)
        res = res.transpose(-1,-2) # (nbatch, na, ncols)
        return res

def rootfinder_solve(alg, A, params, B, E=None, M=None, mparams=[], **options):
    # using rootfinder algorithm
    with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
        nr = A.shape[-1]
        ncols = B.shape[-1]

        # set up the function for the rootfinding
        def fcn_rootfinder(xi):
            # xi: (*BX, nr*ncols)
            x = xi.reshape(*xi.shape[:-1], nr, ncols) # (*BX, nr, ncols)
            y = A.mm(x) - B # (*BX, nr, ncols)
            if E is not None:
                MX = M.mm(x) if M is not None else x
                MXE = MX * E.unsqueeze(-2)
                y = y - MXE # (*BX, nr, ncols)
            y = y.reshape(*xi.shape[:-1], -1) # (*BX, nr*ncols)
            return y

        # setup the initial guess (the batch dimension must be the largest)
        batchdims = _get_batchdims(A, B, E, M)
        x0 = torch.zeros((*batchdims, nr*ncols), dtype=A.dtype, device=A.device)

        if alg == "lbfgs":
            x = lbfgs(fcn_rootfinder, x0, **options)
        elif alg == "broyden":
            x = broyden(fcn_rootfinder, x0, **options)
        else:
            raise RuntimeError("Unknown method %s" % alg)
        x = x.reshape(*x.shape[:-1], nr, ncols)
        return x

def custom_exactsolve(A, params, B, E=None,
                M=None, mparams=[], **options):
    # A: (*BA, na, na)
    # B: (*BB, na, ncols)
    # E: (*BE, ncols)
    # M: (*BM, na, na)
    with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
        return exactsolve(A, B, E, M)

def exactsolve(A:LinearOperator, B:torch.Tensor,
               E:Union[torch.Tensor,None],
               M:Union[LinearOperator,None]):
    # A: (*BA, na, na)
    # B: (*BB, na, ncols)
    # E: (*BE, ncols)
    # M: (*BM, na, na)
    if E is None:
        Amatrix = A.fullmatrix() # (*BA, na, na)
        x, _ = torch.solve(B, Amatrix) # (*BAB, na, ncols)
    elif M is None:
        Amatrix = A.fullmatrix()
        x = _solve_ABE(Amatrix, B, E)
    else:
        Mmatrix = M.fullmatrix() # (*BM, na, na)
        L = torch.cholesky(Mmatrix, upper=False) # (*BM, na, na)
        Linv = torch.inverse(L) # (*BM, na, na)
        LinvT = Linv.transpose(-2,-1) # (*BM, na, na)
        A2 = torch.matmul(Linv, A.mm(LinvT)) # (*BAM, na, na)
        B2 = torch.matmul(Linv, B) # (*BBM, na, ncols)

        X2 = _solve_ABE(A2, B2, E) # (*BABEM, na, ncols)
        x = torch.matmul(LinvT, X2) # (*BABEM, na, ncols)
    return x

def _solve_ABE(A:torch.Tensor, B:torch.Tensor, E:torch.Tensor):
    # A: (*BA, na, na) matrix
    # B: (*BB, na, ncols) matrix
    # E: (*BE, ncols) matrix
    na = A.shape[-1]
    BA, BB, BE = normalize_bcast_dims(A.shape[:-2], B.shape[:-2], E.shape[:-1])
    E = E.view(1, *BE, E.shape[-1]).transpose(0,-1) # (ncols, *BE, 1)
    B = B.view(1, *BB, *B.shape[-2:]).transpose(0,-1) # (ncols, *BB, na, 1)

    # NOTE: The line below is very inefficient for large na and ncols
    AE = A - torch.diag_embed(E.repeat_interleave(repeats=na, dim=-1), dim1=-2, dim2=-1) # (ncols, *BAE, na, na)
    r, _ = torch.solve(B, AE) # (ncols, *BAEM, na, 1)
    r = r.transpose(0,-1).squeeze(0) # (*BAEM, na, ncols)
    return r

def _get_batchdims(A:LinearOperator, B:torch.Tensor, E:Union[torch.Tensor,None], M:Union[LinearOperator,None]):
    batchdims = [A.shape[:-2], B.shape[:-2]]
    if E is not None:
        batchdims.append(E.shape[:-1])
        if M is not None:
            batchdims.append(M.shape[:-2])
    return get_bcasted_dims(*batchdims)
