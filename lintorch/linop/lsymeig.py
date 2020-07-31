import torch
from typing import Union, Mapping, Any
from lintorch.linop.base import LinearOperator
from lintorch.utils.debug import assert_runtime

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

    if "method" not in fwd_options or fwd_options["method"].lower() == "exacteig":
        return exacteig(A, neig, M)
    else:
        raise RuntimeError("Method other than exacteig has not been implemented")

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
