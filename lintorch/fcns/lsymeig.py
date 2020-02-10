import torch
from lintorch.utils.misc import set_default_option
from lintorch.fcns.solve import solve
from lintorch.utils.tensor import tallqr, to_fortran_order
from lintorch.utils.eig import eig

"""
This file contains methods to obtain eigenpairs of a linear transformation
    which is a subclass of ddft.modules.base_linear.BaseLinearModule
"""

__all__ = ["lsymeig"]

def lsymeig(A, params, neig, M=None, mparams=[], fwd_options={}, bck_options={}):
    """
    Obtain `neig` lowest eigenvalues and eigenvectors of a large matrix.
    If M is specified, it solve the eigendecomposition Ax = eMx.

    Arguments
    ---------
    * A: lintorch.Module instance
        The linear module object on which the eigenpairs are constructed.
    * params: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to A.forward(x,*params).
        Each of params must have shape of (nbatch,...)
    * neig: int
        The number of eigenpairs to be retrieved.
    * M: lintorch.Module or None
        The transformation on the right hand side. If None, then M=I.
    * mparams: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to M.
    * fwd_options:
        Eigendecomposition iterative algorithm options.
    * bck_options:
        Conjugate gradient options to calculate the gradient in
        backpropagation calculation.

    Returns
    -------
    * eigvals: (nbatch, neig)
    * eigvecs: (nbatch, na, neig)
        The lowest eigenvalues and eigenvectors.
    """
    na = len(params)
    return lsymeig_torchfcn.apply(A, neig, M, fwd_options, bck_options, na, *params, *mparams)

class lsymeig_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, neig, M, fwd_options, bck_options, na, *amparams):
        # separate the sets of parameters
        params = amparams[:na]
        mparams = amparams[na:]

        config = set_default_option({
            "method": "davidson",
        }, fwd_options)
        ctx.bck_config = set_default_option({
            "min_eps": 1e-8,
        }, bck_options)

        method = config["method"].lower()
        if method == "davidson":
            evals, evecs = davidson(A, params, neig, M, mparams, **config)
        elif method == "exacteig":
            evals, evecs = exacteig(A, params, neig, M, mparams, **config)
        else:
            raise RuntimeError("Unknown eigen decomposition method: %s" % config["method"])

        # save for the backward
        ctx.evals = evals # (nbatch, neig)
        ctx.evecs = evecs # (nbatch, na, neig)
        ctx.params = params
        ctx.A = A
        ctx.M = M
        ctx.mparams = mparams
        return evals, evecs

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        # grad_evals: (nbatch, neig)
        # grad_evecs: (nbatch, na, neig)

        # get the variables from ctx
        evals = ctx.evals
        evecs = ctx.evecs
        M = ctx.M

        # the loss function where the gradient will be retrieved
        params = [p.clone().detach().requires_grad_() for p in ctx.params]
        with torch.enable_grad():
            loss = ctx.A(evecs, *params) # (nbatch, na, neig)

        # calculate the contributions from the eigenvalues
        gevalsA = grad_evals.unsqueeze(1) * evecs # (nbatch, na, neig)

        # calculate the contributions from the eigenvectors
        # orthogonalize the grad_evecs with evecs
        B = _ortho(grad_evecs, evecs, dim=1, M=M, mparams=ctx.mparams)
        gevecs = solve(ctx.A, ctx.params, -B,
            biases=evals, M=M, mparams=ctx.mparams,
            fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
        # orthogonalize gevecs w.r.t. evecs
        gevecsA = _ortho(gevecs, evecs, dim=1, M=M, mparams=ctx.mparams)

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
            mparams = [p.clone().detach().requires_grad_() for p in ctx.mparams]
            with torch.enable_grad():
                mloss = ctx.M(evecs, *mparams) # (nbatch, na, neig)
            gevalsM = -gevalsA * evals.unsqueeze(1)
            gevecsM = -gevecsA * evals.unsqueeze(1)
            gaccumM = gevalsM + gevecsM
            grad_mparams = torch.autograd.grad(
                outputs=(mloss,),
                inputs=mparams,
                grad_outputs=(gaccumM,),
                create_graph=torch.is_grad_enabled(),
            )

        return (None, None, None, None, None, None, *grad_params, *grad_mparams)

def _ortho(A, B, dim=1, M=None, mparams=[]):
    if M is None:
        return A - (A * B).sum(dim=dim, keepdim=True) * B
    else:
        return A - (M(A, *mparams) * B).sum(dim=dim, keepdim=True) * B

def davidson(A, params, neig, M=None, mparams=[], **options):
    """
    Iterative methods to obtain the `neig` lowest eigenvalues and eigenvectors.
    This function is written so that the backpropagation can be done.

    Arguments
    ---------
    * A: BaseLinearModule instance
        The linear module object on which the eigenpairs are constructed.
    * params: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to A.forward(x,*params).
        Each of params must have shape of (nbatch,...)
    * neig: int
        The number of eigenpairs to be retrieved.
    * M: lintorch.Module or None
        The transformation on the right hand side. If None, then M=I.
    * mparams: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to M.
    * **options:
        Iterative algorithm options.

    Returns
    -------
    * eigvals: torch.tensor (nbatch, neig)
    * eigvecs: torch.tensor (nbatch, na, neig)
        The `neig` smallest eigenpairs
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
    na = _check_and_get_shape(A)
    nbatch = params[0].shape[0]
    dtype, device = _get_dtype_device(params, A)

    # set up the initial guess
    V = _set_initial_v(config["v_init"].lower(), dtype, device,
        nbatch, na, nguess,
        M=M, mparams=mparams) # (nbatch,na,nguess)

    prev_eigvals = None
    stop_reason = "max_niter"
    idx = torch.arange(neig).unsqueeze(0).unsqueeze(-1) # (1, neig, 1)
    prev_eigvalT = None
    AV = A(V, *params)
    shift_is_eigvalT = False

    # estimating the lowest eigenvalues
    ntest = 20
    x = torch.randn(nbatch, na, ntest).to(V.dtype).to(V.device) # (nbatch, na, ntest)
    x = x / x.norm(dim=1, keepdim=True)
    Ax = A(x, *params) # (nbatch, na, ntest)
    xTAx = (x * Ax).sum(dim=1) # (nbatch, ntest)
    mean_eig = xTAx.mean(dim=-1) # (nbatch,)
    std_f = (xTAx).std(dim=-1) # (nbatch,)
    std_x2 = (x*x).std()
    rms_eig = (std_f / std_x2) / (na**0.5)
    min_eig_est = mean_eig - 2*rms_eig # (nbatch,)
    min_eig_est = min_eig_est.unsqueeze(-1).repeat(1,neig) # (nbatch,neig)

    for i in range(max_niter):
        VT = V.transpose(-2,-1) # (nbatch,nguess,na)
        # Can be optimized by saving AV from the previous iteration and only
        # operate AV for the new V. This works because the old V has already
        # been orthogonalized, so it will stay the same
        # AV = A(V, *params) # (nbatch,na,nguess)
        T = torch.bmm(VT, AV) # (nbatch,nguess,nguess)

        # eigvals are sorted from the lowest
        # eval: (nbatch,nguess), evec: (nbatch, nguess, nguess)
        eigvalT, eigvecT = torch.symeig(T, eigenvectors=True)
        eigvalT = eigvalT[:,:neig] # (nbatch,neig)
        eigvecT = eigvecT[:,:,:neig] # (nbatch,nguess,neig)

        # calculate the eigenvectors of A
        eigvecA = torch.bmm(V, eigvecT) # (nbatch, na, neig)

        # calculate the residual
        AVs = torch.bmm(AV, eigvecT)
        LVs = eigvalT.unsqueeze(1) * eigvecA # (nbatch, na, neig)
        if M is not None:
            LVs = M(LVs, *mparams)
        resid = AVs - LVs

        # print information and check convergence
        if prev_eigvalT is not None:
            deigval = eigvalT - prev_eigvalT
            max_deigval = deigval.abs().max()
            max_resid = resid.abs().max()
            if verbose:
                print("Iter %3d (guess size: %d): resid: %.3e, devals: %.3e" % \
                      (i+1, nguess, max_resid, max_deigval))
            if max_resid < min_eps:
                break
        if AV.shape[-1] == AV.shape[1]:
            break
        prev_eigvalT = eigvalT

        # apply the preconditioner
        # initial guess of the eigenvalues are actually help really much
        if not shift_is_eigvalT:
            z = min_eig_est # (nbatch,neig)
        else:
            z = eigvalT
        if A.is_precond_set():
            t = A.precond(-resid, *params, biases=z, M=M, mparams=mparams) # (nbatch, na, neig)
        else:
            t = -resid

        # set the estimate of the eigenvalues
        if not shift_is_eigvalT:
            eigvalT_pred = eigvalT + (eigvecA * A(t, *params)).sum(dim=1)
            diff_eigvalT = (eigvalT - eigvalT_pred) # (nbatch, neig)
            if diff_eigvalT.abs().max() < rms_eig*1e-2:
                shift_is_eigvalT = True
            else:
                change_idx = min_eig_est > eigvalT
                next_value = eigvalT - 2*diff_eigvalT
                min_eig_est[change_idx] = next_value[change_idx]

        # orthogonalize t with the rest of the V
        t = to_fortran_order(t)
        Vnew = torch.cat((V, t), dim=-1)
        if Vnew.shape[-1] > Vnew.shape[1]:
            Vnew = Vnew[:,:,:Vnew.shape[1]]
        nadd = Vnew.shape[-1]-V.shape[-1]
        nguess = nguess + nadd
        if M is not None:
            V, R = tallqr(Vnew, MV=M(Vnew, *mparams))
        else:
            V, R = tallqr(Vnew)
        # V, R = torch.qr(Vnew) # (nbatch, na, nguess+neig)
        AVnew = A(V[:,:,-nadd:], *params) # (nbatch,na,nadd)
        AVnew = to_fortran_order(AVnew)
        AV = torch.cat((AV, AVnew), dim=-1)

    eigvals = eigvalT # (nbatch, neig)
    eigvecs = eigvecA # (nbatch, na, neig)
    return eigvals, eigvecs

def exacteig(A, params, neig, M=None, mparams=[], **options):
    """
    The exact method to obtain the `neig` lowest eigenvalues and eigenvectors.
    This function is written so that the backpropagation can be done.

    Arguments
    ---------
    * A: BaseLinearModule instance
        The linear module object on which the eigenpairs are constructed.
    * params: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to A.forward(x,*params).
        Each of params must have shape of (nbatch,...)
    * neig: int
        The number of eigenpairs to be retrieved.
    * M: lintorch.Module or None
        The transformation on the right hand side. If None, then M=I.
    * mparams: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to M.
    * **options:
        The algorithm options.

    Returns
    -------
    * eigvals: torch.tensor (nbatch, neig)
    * eigvecs: torch.tensor (nbatch, na, neig)
        The `neig` smallest eigenpairs
    """
    # obtain the full matrices
    Amatrix = A.fullmatrix(*params)
    if M is None:
        evals, evecs = torch.symeig(Amatrix, eigenvectors=True)
        return evals[:,:neig], evecs[:,:,:neig]
    else:
        Mmatrix = M.fullmatrix(*mparams)
        MA,_ = torch.solve(Amatrix, Mmatrix)
        evals, evecs = eig.apply(MA)
        evals = evals[:,:neig]
        evecs = evecs[:,:,:neig]

        # normalize in M-space
        UMU = torch.sqrt((evecs * torch.matmul(Mmatrix, evecs)).sum(dim=-2, keepdim=True))
        evecs = evecs / UMU
        return evals, evecs


def _get_dtype_device(params, A):
    A_params = list(A.parameters())
    if len(A_params) == 0:
        p = params[0]
    else:
        p = A_params[0]
    dtype = p.dtype
    device = p.device
    return dtype, device

def _check_and_get_shape(A):
    na, nc = A.shape
    if na != nc:
        raise TypeError("The linear transformation of davidson method must be a square matrix")
    return na

def _set_initial_v(vinit_type, dtype, device, nbatch, na, nguess,
                   M=None, mparams=None):
    torch.manual_seed(12421)
    if vinit_type == "eye":
        V = torch.eye(na, nguess).unsqueeze(0).repeat(nbatch,1,1)
    elif vinit_type == "randn":
        V = torch.randn(nbatch, na, nguess)
    elif vinit_type == "random" or vinit_type == "rand":
        V = torch.rand(nbatch, na, nguess)
    else:
        raise ValueError("Unknown v_init type: %s" % vinit_type)

    V = V.to(dtype).to(device)
    # orthogonalize V
    if M is not None:
        V, R = tallqr(V, MV=M(V, *mparams))
    else:
        V, R = tallqr(V)
    return V

if __name__ == "__main__":
    import time
    from lintorch.utils.fd import finite_differences
    from lintorch.core.base import Module

    # generate the matrix
    na = 20
    dtype = torch.float64
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).unsqueeze(0).requires_grad_(True)
    M1 = (torch.rand((1,na,na))*0.1).to(dtype).requires_grad_(True)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).unsqueeze(0).requires_grad_(True)

    class Acls(Module):
        def __init__(self):
            super(Acls, self).__init__(shape=(na,na))

        def forward(self, x, A1, diag):
            Amatrix = (A1 + A1.transpose(-2,-1))
            A = Amatrix + diag.diag_embed(dim1=-2, dim2=-1)
            y = torch.bmm(A, x)
            return y

        def precond(self, y, A1, dg, biases=None, M=None, mparams=None):
            # return y
            # y: (nbatch, na, ncols)
            # dg: (nbatch, na)
            # biases: (nbatch, ncols) or None
            Adiag = A1.diagonal(dim1=-2, dim2=-1) * 2
            dd = (Adiag + dg).unsqueeze(-1)

            if biases is not None:
                if M is not None:
                    M1, Mdg = mparams
                    Mdiag = M1.diagonal(dim1=-2, dim2=-1) * 2
                    md = (Mdiag + Mdg).unsqueeze(-1)
                    dd = dd - biases.unsqueeze(1) * md
                else:
                    dd = dd - biases.unsqueeze(1) # (nbatch, na, ncols)
            dd[dd.abs() < 1e-6] = 1.0
            yprec = y / dd
            return yprec

    def getloss(A1, diag, M1, mdiag):
        A = Acls()
        M = Acls()
        neig = 4
        options = {
            "method": "exacteig",
            "verbose": False,
            "nguess": neig,
            "v_init": "randn",
        }
        bck_options = {
            "verbose": True,
            "min_eps": 1e-9,
        }
        with torch.enable_grad():
            A1.requires_grad_()
            diag.requires_grad_()
            M1.requires_grad_()
            mdiag.requires_grad_()
            evals, evecs = lsymeig(A,
                neig=neig,
                params=(A1, diag,),
                M=M,
                mparams=(M1, mdiag,),
                fwd_options=options,
                bck_options=bck_options)

            lss = 0
            lss = lss + (evals**1).abs().sum() # correct
            # lss = lss + (evecs**3).abs().sum()
            # grad_A1, grad_diag = torch.autograd.grad(lss, (A1, diag),
            #     create_graph=True)

        loss = 0
        loss = loss + lss
        # loss = loss + (grad_A1**2).abs().sum()
        # loss = loss + (grad_diag**2).abs().sum()
        return loss

    t0 = time.time()
    loss = getloss(A1, diag, M1, mdiag)
    t1 = time.time()
    print("Forward done in %fs" % (t1 - t0))
    loss.backward()
    t2 = time.time()
    print("Backward done in %fs" % (t2 - t1))
    Agrad = A1.grad.data
    dgrad = diag.grad.data
    Mgrad = M1.grad.data
    mdgrad = mdiag.grad.data

    Afd = finite_differences(getloss, (A1, diag, M1, mdiag), 0, eps=1e-4)
    dfd = finite_differences(getloss, (A1, diag, M1, mdiag), 1, eps=1e-4)
    Mfd = finite_differences(getloss, (A1, diag, M1, mdiag), 2, eps=1e-4)
    mdfd = finite_differences(getloss, (A1, diag, M1, mdiag), 3, eps=1e-4)
    print("Finite differences done")

    print("A1:")
    print(Agrad)
    print(Afd)
    print(Agrad/Afd)

    print("diag:")
    print(dgrad)
    print(dfd)
    print(dgrad/dfd)

    print("M1:")
    print(Mgrad)
    print(Mfd)
    print(Mgrad/Mfd)

    print("mdiag:")
    print(mdgrad)
    print(mdfd)
    print(mdgrad/mdfd)
