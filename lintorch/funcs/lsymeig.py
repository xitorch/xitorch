import torch
from typing import Union, Mapping, Any
import functools
from lintorch.core.editable_module import wrap_fcn
from lintorch.core.linop import LinearOperator
from lintorch.funcs.solve import solve
from lintorch.utils.assertfuncs import assert_runtime
from lintorch.utils.debugmodes import is_debug_enabled
from lintorch.utils.bcast import get_bcasted_dims
from lintorch.utils.misc import set_default_option, dummy_context_manager
from lintorch.utils.tensor import tallqr, to_fortran_order, ortho

def lsymeig(A:LinearOperator, neig:Union[int,None]=None,
        M:Union[LinearOperator,None]=None,
        fwd_options:Mapping[str,Any]={}, bck_options:Mapping[str,Any]={}):
    """
    Obtain `neig` lowest eigenvalues and eigenvectors of a linear operator.
    If M is specified, it solve the eigendecomposition Ax = eMx.

    Arguments
    ---------
    * A: lintorch.LinearOperator hermitian instance with shape (*BA, q, q)
        The linear module object on which the eigenpairs are constructed.
    * neig: int or None
        The number of eigenpairs to be retrieved. If None, all eigenpairs are
        retrieved
    * M: lintorch.LinearOperator hermitian instance with shape (*BM, q, q) or None
        The transformation on the right hand side. If None, then M=I.
    * fwd_options: dict with str as key
        Eigendecomposition iterative algorithm options.
    * bck_options: dict with str as key
        Conjugate gradient options to calculate the gradient in
        backpropagation calculation.

    Returns
    -------
    * eigvals: (*BAM, neig)
    * eigvecs: (*BAM, na, neig)
        The lowest eigenvalues and eigenvectors, where *BAM are the broadcasted
        shape of *BA and *BM.
    """
    assert_runtime(A.is_hermitian, "The linear operator A must be Hermitian")
    if M is not None:
        assert_runtime(M.is_hermitian, "The linear operator M must be Hermitian")
        assert_runtime(M.shape[-1] == A.shape[-1], "The shape of A & M must match (A: %s, M: %s)" % (A.shape, M.shape))

    # perform expensive check if debug mode is enabled
    if is_debug_enabled():
        A.check()
        if M is not None:
            M.check()

    if "method" not in fwd_options or fwd_options["method"].lower() == "exacteig":
        return exacteig(A, neig, M)
    else:
        # get the unique parameters of A & M
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return lsymeig_torchfcn.apply(A, neig, M,
            fwd_options, bck_options,
            na, *params, *mparams)

def exacteig(A:LinearOperator, neig:Union[int,None], M:Union[LinearOperator,None]):
    Amatrix = A.fullmatrix() # (*BA, q, q)
    if neig is None:
        neig = A.shape[-1]
    if M is None:
        evals, evecs = torch.symeig(Amatrix, eigenvectors=True) # (*BA, q), (*BA, q, q)
        return evals[...,:neig], evecs[...,:neig]
    else:
        Mmatrix = M.fullmatrix() # (*BM, q, q)

        # M decomposition to make A symmetric
        # it is done this way to make it numerically stable in avoiding
        # complex eigenvalues for (near-)degenerate case
        L = torch.cholesky(Mmatrix, upper=False) # (*BM, q, q)
        Linv = torch.inverse(L) # (*BM, q, q)
        LinvT = Linv.transpose(-2,-1) # (*BM, q, q)
        A2 = torch.matmul(Linv, torch.matmul(Amatrix, LinvT)) # (*BAM, q, q)

        # calculate the eigenvalues and eigenvectors
        # (the eigvecs are normalized in M-space)
        evals, evecs = torch.symeig(A2, eigenvectors=True) # (*BAM, q, q)
        evals = evals[...,:neig] # (*BAM, neig)
        evecs = evecs[...,:neig] # (*BAM, q, neig)
        evecs = torch.matmul(LinvT, evecs)
        return evals, evecs

class lsymeig_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, neig, M, fwd_options, bck_options, na, *amparams):
        # A: LinearOperator (*BA, q, q)
        # M: LinearOperator (*BM, q, q) or None

        # separate the sets of parameters
        params = amparams[:na]
        mparams = amparams[na:]

        config = set_default_option({
        }, fwd_options)
        ctx.bck_config = set_default_option({
            # "method": ???
        }, bck_options)

        method = config["method"].lower()
        if method == "davidson":
            evals, evecs = davidson(A, params, neig, M, mparams, **config)
        else:
            raise RuntimeError("Unknown eigen decomposition method: %s" % config["method"])

        # save for the backward
        ctx.evals = evals # (*BAM, neig)
        ctx.evecs = evecs # (*BAM, na, neig)
        ctx.params = params
        ctx.A = A
        ctx.M = M
        ctx.mparams = mparams
        return evals, evecs

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        # grad_evals: (*BAM, neig)
        # grad_evecs: (*BAM, na, neig)

        # get the variables from ctx
        evals = ctx.evals
        evecs = ctx.evecs
        M = ctx.M
        A = ctx.A

        # the loss function where the gradient will be retrieved
        # warnings: if not all params have the connection to the output of A,
        # it could cause an infinite loop because pytorch will keep looking
        # for the *params node and propagate further backward via the `evecs`
        # path. So make sure all the *params are all connected in the graph.
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in ctx.params]
            with A.uselinopparams(*params):
                loss = A.mm(evecs) # (*BAM, na, neig)

        # calculate the contributions from the eigenvalues
        gevalsA = grad_evals.unsqueeze(-2) * evecs # (*BAM, na, neig)

        # calculate the contributions from the eigenvectors
        with M.uselinopparams(*ctx.mparams) if M is not None else dummy_context_manager():
            # orthogonalize the grad_evecs with evecs
            B = ortho(grad_evecs, evecs, dim=-2, M=M, mright=False)
            with A.uselinopparams(*ctx.params):
                gevecs = solve(A, -B, evals, M, fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
            # orthogonalize gevecs w.r.t. evecs
            gevecsA = ortho(gevecs, evecs, dim=-2, M=M, mright=True)

        # accummulate the gradient contributions
        gaccumA = gevalsA + gevecsA
        grad_params = torch.autograd.grad(
            outputs=(loss,),
            inputs=params,
            grad_outputs=(gaccumA,),
            create_graph=torch.is_grad_enabled(),
        )

        grad_mparams = []
        if ctx.M is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in ctx.mparams]
                with M.uselinopparams(*mparams):
                    mloss = M.mm(evecs) # (*BAM, na, neig)
            gevalsM = -gevalsA * evals.unsqueeze(-2)
            gevecsM = -gevecsA * evals.unsqueeze(-2)

            # the contribution from the parallel elements
            gevecsM_par = (-0.5 * torch.einsum("...ae,...ae->...e", grad_evecs, evecs)).unsqueeze(-2) * evecs # (*BAM, na, neig)

            gaccumM = gevalsM + gevecsM + gevecsM_par
            grad_mparams = torch.autograd.grad(
                outputs=(mloss,),
                inputs=mparams,
                grad_outputs=(gaccumM,),
                create_graph=torch.is_grad_enabled(),
            )

        return (None, None, None, None, None, None, *grad_params, *grad_mparams)

def davidson(A, params, neig, M=None, mparams=[], **options):
    """
    Iterative methods to obtain the `neig` lowest eigenvalues and eigenvectors.
    This function is written so that the backpropagation can be done.
    It solves the eigendecomposition AV = VME where V are the matrix of eigenvectors,
    and E are the diagonal matrix consists of the eigenvalues.

    Arguments
    ---------
    * A: LinearOperator instance (*BA, na, na)
        The linear operator object on which the eigenpairs are constructed.
    * params: list of differentiable torch.tensor of any shapes
        List of differentiable torch.tensor to be put to A.
    * neig: int
        The number of eigenpairs to be retrieved.
    * M: LinearOperator instance (*BM, na, na) or None
        The transformation on the right hand side. If None, then M=I.
    * mparams: list of differentiable torch.tensor of any shapes
        List of differentiable torch.tensor to be put to M.
    * **options:
        Iterative algorithm options.

    Returns
    -------
    * eigvals: torch.tensor (*BAM, neig)
    * eigvecs: torch.tensor (*BAM, na, neig)
        The `neig` lowest eigenpairs
    """
    config = set_default_option({
        "max_niter": 1000,
        "nguess": neig, # number of initial guess
        "min_eps": 1e-6,
        "verbose": False,
        "eps_cond": 1e-6,
        "v_init": "randn",
        "max_addition": neig,
    }, options)

    # get some of the options
    nguess = config["nguess"]
    max_niter = config["max_niter"]
    min_eps = config["min_eps"]
    verbose = config["verbose"]
    eps_cond = config["eps_cond"]
    max_addition = config["max_addition"]

    # get the shape of the transformation
    na = A.shape[-1]
    if M is None:
        bcast_dims = A.shape[:-2]
    else:
        bcast_dims = get_bcasted_dims(A.shape[:-2], M.shape[:-2])
    dtype = A.dtype
    device = A.device

    # TODO: A to use params
    prev_eigvals = None
    prev_eigvalT = None
    stop_reason = "max_niter"
    shift_is_eigvalT = False
    idx = torch.arange(neig).unsqueeze(-1) # (neig, 1)

    with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
        # set up the initial guess
        V = _set_initial_v(config["v_init"].lower(), dtype, device,
            bcast_dims, na, nguess,
            M=M, mparams=mparams) # (*BAM, na, nguess)
        # V = V.reshape(*bcast_dims, na, nguess) # (*BAM, na, nguess)

        # estimating the lowest eigenvalues
        min_eig_est, rms_eig = _estimate_lowest_eigvals(A, neig,
            bcast_dims=bcast_dims, na=na, ntest=20,
            dtype=V.dtype, device=V.device)

        best_resid = float("inf")
        AV = A.mm(V)
        for i in range(max_niter):
            VT = V.transpose(-2,-1) # (*BAM,nguess,na)
            # Can be optimized by saving AV from the previous iteration and only
            # operate AV for the new V. This works because the old V has already
            # been orthogonalized, so it will stay the same
            # AV = A.mm(V) # (*BAM,na,nguess)
            T = torch.matmul(VT, AV) # (*BAM,nguess,nguess)

            # eigvals are sorted from the lowest
            # eval: (*BAM, nguess), evec: (*BAM, nguess, nguess)
            eigvalT, eigvecT = torch.symeig(T, eigenvectors=True)
            eigvalT = eigvalT[...,:neig] # (*BAM,neig)
            eigvecT = eigvecT[...,:neig] # (*BAM,nguess,neig)

            # calculate the eigenvectors of A
            eigvecA = torch.matmul(V, eigvecT) # (*BAM, na, neig)

            # calculate the residual
            AVs = torch.matmul(AV, eigvecT) # (*BAM, na, neig)
            LVs = eigvalT.unsqueeze(-2) * eigvecA # (*BAM, na, neig)
            if M is not None:
                LVs = M.mm(LVs)
            resid = AVs - LVs # (*BAM, na, neig)

            # print information and check convergence
            max_resid = resid.abs().max()
            if prev_eigvalT is not None:
                deigval = eigvalT - prev_eigvalT
                max_deigval = deigval.abs().max()
                if verbose:
                    print("Iter %3d (guess size: %d): resid: %.3e, devals: %.3e" % \
                          (i+1, nguess, max_resid, max_deigval))

            if max_resid < best_resid:
                best_resid = max_resid
                best_eigvals = eigvalT
                best_eigvecs = eigvecA
            if max_resid < min_eps:
                break
            if AV.shape[-1] == AV.shape[-2]:
                break
            prev_eigvalT = eigvalT

            # apply the preconditioner
            # initial guess of the eigenvalues are actually help really much
            if not shift_is_eigvalT:
                z = min_eig_est # (*BAM,neig)
            else:
                z = eigvalT # (*BAM,neig)
            # if A.is_precond_set():
            #     t = A.precond(-resid, *params, biases=z, M=M, mparams=mparams) # (nbatch, na, neig)
            # else:
            t = -resid # (*BAM, na, neig)

            # set the estimate of the eigenvalues
            if not shift_is_eigvalT:
                eigvalT_pred = eigvalT + torch.einsum('...ae,...ae->...e', eigvecA, A.mm(t)) # (*BAM, neig)
                diff_eigvalT = (eigvalT - eigvalT_pred) # (*BAM, neig)
                if diff_eigvalT.abs().max() < rms_eig*1e-2:
                    shift_is_eigvalT = True
                else:
                    change_idx = min_eig_est > eigvalT
                    next_value = eigvalT - 2*diff_eigvalT
                    min_eig_est[change_idx] = next_value[change_idx]

            # orthogonalize t with the rest of the V
            t = to_fortran_order(t)
            Vnew = torch.cat((V, t), dim=-1)
            if Vnew.shape[-1] > Vnew.shape[-2]:
                Vnew = Vnew[...,:Vnew.shape[-2]]
            nadd = Vnew.shape[-1] - V.shape[-1]
            nguess = nguess + nadd
            if M is not None:
                MV_ = M.mm(Vnew)
                V, R = tallqr(Vnew, MV=MV_)
            else:
                V, R = tallqr(Vnew)
            AVnew = A.mm(V[...,-nadd:]) # (*BAM,na,nadd)
            AVnew = to_fortran_order(AVnew)
            AV = torch.cat((AV, AVnew), dim=-1)

    eigvals = best_eigvals # (*BAM, neig)
    eigvecs = best_eigvecs # (*BAM, na, neig)
    return eigvals, eigvecs

def _set_initial_v(vinit_type, dtype, device, batch_dims, na, nguess,
                   M=None, mparams=None):
    torch.manual_seed(12421)
    if vinit_type == "eye":
        nbatch = functools.reduce(lambda x,y: x*y, bcast_dims, 1)
        V = torch.eye((na, nguess), dtype=dtype, device=device).unsqueeze(0).repeat(nbatch,1,1).reshape(*batch_dims, na, nguess)
    elif vinit_type == "randn":
        V = torch.randn((*batch_dims, na, nguess), dtype=dtype, device=device)
    elif vinit_type == "random" or vinit_type == "rand":
        V = torch.rand((*batch_dims, na, nguess), dtype=dtype, device=device)
    else:
        raise ValueError("Unknown v_init type: %s" % vinit_type)

    # orthogonalize V
    if M is not None:
        V, R = tallqr(V, MV=M.mm(V))
    else:
        V, R = tallqr(V)
    return V

def _estimate_lowest_eigvals(A, neig, bcast_dims, na, ntest, dtype, device):
    # estimate the lowest eigen value
    x = torch.randn((*bcast_dims, na, ntest), dtype=dtype, device=device) # (*BAM, na, ntest)
    x = x / x.norm(dim=-2, keepdim=True)
    Ax = A.mm(x) # (*BAM, na, ntest)
    xTAx = (x * Ax).sum(dim=-2) # (*BAM, ntest)
    mean_eig = xTAx.mean(dim=-1) # (*BAM,)
    std_f = (xTAx).std(dim=-1) # (*BAM,)
    std_x2 = (x*x).std()
    rms_eig = (std_f / std_x2) / (na**0.5) # (*BAM,)
    min_eig_est = mean_eig - 2*rms_eig # (*BAM,)
    min_eig_est = min_eig_est.unsqueeze(-1).repeat_interleave(repeats=neig, dim=-1) # (*BAM,neig)
    return min_eig_est, rms_eig.max()
