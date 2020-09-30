import inspect
from typing import Sequence, Optional, List
import warnings
import traceback
import torch
from abc import abstractmethod, abstractproperty
from contextlib import contextmanager
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from xitorch._core.editable_module import EditableModule
from xitorch.debug.modes import is_debug_enabled
from xitorch._utils.bcast import get_bcasted_dims

__all__ = ["LinearOperator"]

class LinearOperator(EditableModule):
    """
    ``LinearOperator`` is a base class designed to behave as a linear operator
    without explicitly determining the matrix.
    This ``LinearOperator`` should be able to operate as batched linear
    operators where its shape is ``(B1,B2,...,Bb,p,q)``
    with ``B*`` as the (optional) batch dimensions.

    For a user-defined class to behave as ``LinearOperator``, it must use
    ``LinearOperator`` as one of the parent and it has to have ``._mv()``
    and ``._getparamnames()`` methods implemented at the very least.
    """
    @classmethod
    def m(cls, mat:torch.Tensor, is_hermitian:Optional[bool]=None):
        """
        Class method to wrap a matrix into ``LinearOperator``.

        Arguments
        ---------
        mat: torch.Tensor
            Matrix to be wrapped in the ``LinearOperator``.
        is_hermitian: bool or None
            Indicating if the matrix is Hermitian. If ``None``, the symmetry
            will be checked. If supplied as a bool, there is no check performed.

        Returns
        -------
        LinearOperator
            Linear operator object that represents the matrix.

        Example
        -------
        .. testsetup:: *

            import torch
            import xitorch

        .. doctest::

            >>> mat = torch.rand(4,2,5,5)  # 5x5 matrix with (4,2) batch dimensions
            >>> linop = xitorch.LinearOperator.m(mat)
            >>> print(linop)
            xitorch.LinearOperator with shape: (4,2,5,5)
        """
        if is_hermitian is None:
            if mat.shape[-2] != mat.shape[-1]:
                is_hermitian = False
            else:
                is_hermitian = torch.allclose(mat, mat.transpose(-2,-1))

        if is_hermitian:
            return _MatrixHermitLinOp(mat)
        else:
            return _MatrixNonHermitLinOp(mat)

    def __init__(self, shape:Sequence[int],
            is_hermitian:bool = False,
            dtype:Optional[torch.dtype] = None,
            device:Optional[torch.device] = None,
            _suppress_hermit_warning:bool = False) -> None:

        super(LinearOperator, self).__init__()
        if len(shape) < 2:
            raise RuntimeError("The shape must have at least 2 dimensions")
        self._shape = shape
        self._batchshape = list(shape[:-2])
        self._is_hermitian = is_hermitian
        self._dtype = dtype if dtype is not None else torch.float32
        self._device = device if device is not None else torch.device("cpu")
        if is_hermitian and shape[-1] != shape[-2]:
            raise RuntimeError("The object is indicated as Hermitian, but the shape is not square")

        # check which methods are implemented
        self._is_mv_implemented = self.__check_if_implemented("_mv")
        self._is_mm_implemented = self.__check_if_implemented("_mm")
        self._is_rmv_implemented = self.__check_if_implemented("_rmv")
        self._is_rmm_implemented = self.__check_if_implemented("_rmm")
        self._is_fullmatrix_implemented = self.__check_if_implemented("_fullmatrix")
        if not self._is_mv_implemented:
            raise RuntimeError("LinearOperator must have at least ._mv() method implemented")
        if not _suppress_hermit_warning and self._is_hermitian and (self._is_rmv_implemented or self._is_rmm_implemented):
            warnings.warn("The LinearOperator is Hermitian with implemented rmv or rmm. We will use the mv and mm methods instead")

        # caches
        self._matrix_defined = False
        self._matrix = torch.tensor([])

    def __repr__(self) -> str:
        return "xitorch.LinearOperator with shape: (%s)" % \
            (",".join([str(s) for s in self.shape]))

    @abstractmethod
    def _getparamnames(self, prefix:str="") -> List[str]:
        """
        List the self's parameters that affecting the ``LinearOperator``.
        This is for the derivative purpose.

        Arguments
        ---------
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        list of str
            List of parameter names (including the prefix) that affecting
            the ``LinearOperator``.
        """
        pass

    @abstractmethod
    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for matrix-vector multiplication.
        Required for all ``LinearOperator`` objects.
        """
        pass

    # @abstractmethod
    def _rmv(self, x:torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for transposed matrix-vector multiplication.
        Optional, but will raise an error if not implemented but ``.rmv()``
        is called.
        """
        pass

    # @abstractmethod # (optional)
    def _mm(self, x:torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for matrix-matrix multiplication.
        If not implemented, then it uses batched version of matrix-vector
        multiplication.
        Usually this is implemented for efficiency reasons.
        """
        pass

    # @abstractmethod
    def _rmm(self, x:torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for transposed matrix-matrix multiplication.
        If not implemented, then it uses batched version of transposed matrix-vector
        multiplication.
        Usually this is implemented for efficiency reasons.
        """
        pass

    # @abstractmethod
    def _fullmatrix(self) -> torch.Tensor:
        pass

    # linear operators must have a set of parameters that affects most of
    # the methods (i.e. mm, mv, rmm, rmv)
    def getlinopparams(self) -> Sequence[torch.Tensor]:
        return self.getuniqueparams("mm")

    @contextmanager
    def uselinopparams(self, *params):
        methodname = "mm"
        try:
            _orig_params_ = self.getuniqueparams(methodname)
            self.setuniqueparams(methodname, *params)
            yield self
        except Exception as exc:
            raise exc
            # traceback.print_exc()
        finally:
            self.setuniqueparams(methodname, *_orig_params_)

    ############# implemented functions ################
    def mv(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-vector operation to vector ``x`` with shape ``(...,q)``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Arguments
        ---------
        x: torch.tensor
            The vector with shape ``(...,q)`` where the linear operation is operated on

        Returns
        -------
        y: torch.tensor
            The result of the linear operation with shape ``(...,p)``
        """
        if x.shape[-1] != self.shape[-1]:
            raise RuntimeError("Cannot operate .mv on shape %s. Expected (...,%d)" %\
                (str(tuple(x.shape)), self.shape[-1]))

        return self._mv(x)

    def mm(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-matrix operation to matrix ``x`` with shape ``(...,q,r)``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Arguments
        ---------
        x: torch.tensor
            The matrix with shape ``(...,q,r)`` where the linear operation is operated on.

        Returns
        -------
        y: torch.tensor
            The result of the linear operation with shape ``(...,p,r)``
        """
        if x.shape[-2] != self.shape[-1]:
            raise RuntimeError("Cannot operate .mm on shape %s. Expected (...,%d,*)" %\
                (str(tuple(x.shape)), self.shape[-1]))

        xbatchshape = list(x.shape[:-2])
        if self._is_mm_implemented:
            return self._mm(x)
        else:
            # use batched mv as mm

            # move the last dimension to the very first dimension to be broadcasted
            if len(xbatchshape) < len(self._batchshape):
                xbatchshape = [1]*(len(self._batchshape)-len(xbatchshape)) + xbatchshape
            x1 = x.view(1, *xbatchshape, *x.shape[-2:])
            xnew = x1.transpose(0, -1).squeeze(-1) # (r,...,q)

            # apply batched mv and restore the initial shape
            ynew = self._mv(xnew) # (r,...,p)
            y = ynew.unsqueeze(-1).transpose(0,-1).squeeze(0) # (...,p,r)
            return y

    def rmv(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-vector adjoint operation to vector ``x`` with shape ``(...,p)``,
        i.e. ``A^H x``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Arguments
        ---------
        x: torch.tensor
            The vector of shape ``(...,p)`` where the adjoint linear operation is operated at.

        Returns
        -------
        y: torch.tensor
            The result of the adjoint linear operation with shape ``(...,q)``
        """
        if x.shape[-1] != self.shape[-2]:
            raise RuntimeError("Cannot operate .rmv on shape %s. Expected (...,%d)" %\
                (str(tuple(x.shape)), self.shape[-2]))

        if self._is_hermitian:
            return self._mv(x)
        elif not self._is_rmv_implemented:
            raise RuntimeError("The ._rmv() must be implemented to call .rmv() method")
        return self._rmv(x)

    def rmm(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-matrix adjoint operation to matrix ``x`` with shape ``(...,p,r)``,
        i.e. ``A^H X``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Arguments
        ---------
        x: torch.Tensor
            The matrix of shape ``(...,p,r)`` where the adjoint linear operation is operated on.

        Returns
        -------
        y: torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q,r)``.
        """
        if x.shape[-2] != self.shape[-2]:
            raise RuntimeError("Cannot operate .rmm on shape %s. Expected (...,%d,*)" %\
                (str(tuple(x.shape)), self.shape[-2]))

        if self._is_hermitian:
            return self.mm(x)

        xbatchshape = list(x.shape[:-2])
        if self._is_rmm_implemented:
            return self._rmm(x)
        elif not self._is_rmv_implemented:
            raise RuntimeError("The ._rmv() or ._rmm() must be implemented to call .rmm() method")
        else:
            # use batched mv as mm

            # move the last dimension to the very first dimension to be broadcasted
            if len(xbatchshape) < len(self._batchshape):
                xbatchshape = [1]*(len(self._batchshape)-len(xbatchshape)) + xbatchshape
            x1 = x.view(1, *xbatchshape, *x.shape[-2:]) # (1,...,p,r)
            xnew = x1.transpose(0, -1).squeeze(-1) # (r,...,p)

            # apply batched mv and restore the initial shape
            ynew = self._rmv(xnew) # (r,...,q)
            y = ynew.unsqueeze(-1).transpose(0,-1).squeeze(0) # (...,q,r)
            return y

    def fullmatrix(self) -> torch.Tensor:
        if not self._matrix_defined:
            if self._is_fullmatrix_implemented:
                self._matrix = self._fullmatrix()
            else:
                nq = self._shape[-1]
                V = torch.eye(nq, dtype=self._dtype, device=self._device) # (nq,nq)
                self._matrix = self.mm(V) # (B1,B2,...,Bb,np,nq)
            self._matrix_defined = True
        return self._matrix

    def scipy_linalg_op(self):
        to_tensor = lambda x: torch.tensor(x, dtype=self.dtype, device=self.device)
        return spLinearOperator(
            shape=self.shape,
            matvec =lambda v: self.mv (to_tensor(v)).detach().cpu().numpy(),
            rmatvec=lambda v: self.rmv(to_tensor(v)).detach().cpu().numpy(),
            matmat =lambda v: self.mm (to_tensor(v)).detach().cpu().numpy(),
            rmatmat=lambda v: self.rmm(to_tensor(v)).detach().cpu().numpy(),
        )

    def getparamnames(self, methodname:str, prefix:str="") -> List[str]:
        """"""
        # just to remove the docstring from EditableModule because user
        # does not need to know about this function

        if methodname in ["mv", "rmv", "mm", "rmm"]:
            return self._getparamnames(prefix=prefix)
        elif methodname == "fullmatrix":
            return [prefix+"_matrix"]
        else:
            raise KeyError("getparamnames for method %s is not implemented" % methodname)

    ############# cached properties ################
    @property
    def H(self):
        return AdjointLinearOperator(self)

    ############# special functions ################
    def matmul(self, b, is_hermitian:bool=False):
        # returns linear operator that represents self @ b
        if self.shape[-1] != b.shape[-2]:
            raise RuntimeError("Mismatch shape of matmul operation: %s and %s" % (self.shape, b.shape))
        return MatmulLinearOp(self, b, is_hermitian=is_hermitian)

    ############# properties ################
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    # implementation
    @property
    def is_mv_implemented(self) -> bool:
        return self._is_mv_implemented

    @property
    def is_mm_implemented(self) -> bool:
        return self._is_mm_implemented

    @property
    def is_rmv_implemented(self) -> bool:
        return self._is_rmv_implemented

    @property
    def is_rmm_implemented(self) -> bool:
        return self._is_rmm_implemented

    @property
    def is_fullmatrix_implemented(self) -> bool:
        return self._is_fullmatrix_implemented

    ############ debug functions ##############
    def check(self, warn:Optional[bool]=None) -> None:
        """
        Perform checks to make sure the ``LinearOperator`` behaves as a proper
        linear operator.

        Arguments
        ---------
        warn: bool or None
            If ``True``, then raises a warning to the user that the check might slow
            down the program. This is to remind the user to turn off the check
            when not in a debugging mode.
            If ``None``, it will raise a warning if it runs not in a debug mode, but
            will be silent if it runs in a debug mode.

        Raises
        ------
        RuntimeError
            Raised if an error is raised when performing linear operations of the
            object (e.g. calling ``.mv()``, ``.mm()``, etc)
        AssertionError
            Raised if the linear operations do not behave as proper linear operations.
            (e.g. not scaling linearly)
        """
        if warn is None:
            warn = not is_debug_enabled()
        if warn:
            msg = "The linear operator check is performed. This might slow down your program."
            warnings.warn(msg, stacklevel=2)
        checklinop(self)
        print("Check linear operator done")

    ############ private functions #################
    def __check_if_implemented(self, methodname:str) -> bool:
        this_method = getattr(self, methodname).__func__
        base_method = getattr(LinearOperator, methodname)
        return this_method is not base_method

############## special linear operators ##############
class AdjointLinearOperator(LinearOperator):
    def __init__(self, obj:LinearOperator):
        super(AdjointLinearOperator, self).__init__(
            shape = (*obj.shape[:-2], obj.shape[-1], obj.shape[-2]),
            is_hermitian = obj.is_hermitian,
            dtype = obj.dtype,
            device = obj.device
        )
        self.obj = obj

    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        if not self.obj.is_rmv_implemented:
            raise RuntimeError("The ._rmv of must be implemented to call .H.mv()")
        return self.obj._rmv(x)

    def _rmv(self, x:torch.Tensor) -> torch.Tensor:
        return self.obj._mv(x)

    def _getparamnames(self, prefix:str="") -> List[str]:
        return self.obj._getparamnames(prefix=prefix+"obj.")

    @property
    def H(self):
        return self.obj

class MatmulLinearOp(LinearOperator):
    def __init__(self, a:LinearOperator, b:LinearOperator, is_hermitian:bool=False):
        shape = (*get_bcasted_dims(a.shape[:-2], b.shape[:-2]), a.shape[-2], b.shape[-1])
        super(MatmulLinearOp, self).__init__(
            shape = shape,
            is_hermitian = is_hermitian,
            dtype = a.dtype,
            device = a.device,
            _suppress_hermit_warning = True,
        )
        self.a = a
        self.b = b

    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        return self.a._mv(self.b._mv(x))

    def _rmv(self, x:torch.Tensor) -> torch.Tensor:
        return self.b.rmv(self.a.rmv(x))

    def _getparamnames(self, prefix:str="") -> List[str]:
        return self.a._getparamnames(prefix=prefix+"a.") + \
               self.b._getparamnames(prefix=prefix+"b.")

# distinguishing the classes of Hermitian and non-Hermitian matrices to suppress
# the warnings about redundant implementation of rmm and rmv
class _MatrixNonHermitLinOp(LinearOperator):
    def __init__(self, mat:torch.Tensor) -> None:

        super(_MatrixNonHermitLinOp, self).__init__(
            shape = mat.shape,
            is_hermitian = False,
            dtype = mat.dtype,
            device = mat.device
        )
        self.mat = mat

    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

    def _mm(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x)

    def _rmv(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat.transpose(-2,-1), x.unsqueeze(-1)).squeeze(-1)

    def _rmm(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat.transpose(-2,-1), x)

    def _fullmatrix(self) -> torch.Tensor:
        return self.mat

    def _getparamnames(self, prefix:str="") -> List[str]:
        return [prefix+"mat"]

class _MatrixHermitLinOp(LinearOperator):
    def __init__(self, mat:torch.Tensor) -> None:
        super(_MatrixHermitLinOp, self).__init__(
            shape = mat.shape,
            is_hermitian = True,
            dtype = mat.dtype,
            device = mat.device
        )
        # make sure the gradient is symmetrically distributed in the matrix
        self.mat = (mat + mat.transpose(-2,-1)) * 0.5

    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

    def _mm(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x)

    def _fullmatrix(self) -> torch.Tensor:
        return self.mat

    def _getparamnames(self, prefix:str="") -> List[str]:
        return [prefix+"mat"]

    @property
    def H(self):
        return self

def checklinop(linop:LinearOperator) -> None:
    """
    Check if the implemented mv and mm can receive the possible shapes and returns
    the correct shape. If an error is found, then this function raise AssertionError.

    Argument
    --------
    * linop: LinearOperator instance
        The instance of LinearOperator to be checked

    Exception
    ---------
    * AssertionError
        Raised if there is a shape mismatch
    * RuntimeError
        Raised if there is an error when evaluating the .mv, .mm, .rmv, or .rmm methods
    """
    shape = linop.shape
    p, q = shape[-2:]
    batchshape = shape[:-2]

    def runtest(methodname, xshape, yshape):
        x = torch.rand(xshape, dtype=linop.dtype, device=linop.device)
        fcn = getattr(linop, methodname)
        try:
            y = fcn(x)
        except:
            s = traceback.format_exc()
            msg = "An error is raised from .%s with input shape: %s (linear operator shape: %s)\n" % \
                (methodname, tuple(xshape), tuple(linop.shape))
            msg += "--- full traceback ---\n%s" % s
            raise RuntimeError(msg)
        msg = "The output shape of .%s is not correct. Input: %s, expected output: %s, output: %s" % \
            (methodname, tuple(x.shape), tuple(yshape), tuple(y.shape))
        assert list(y.shape) == list(yshape), msg

        # linearity test
        x2 = 1.25*x
        y2 = fcn(x2)
        assert torch.allclose(y2, 1.25*y), "Linearity check fails"
        y0 = fcn(0*x)
        assert torch.allclose(y0, y*0), "Linearity check (with 0) fails"

        # batched test
        xnew = torch.cat((x.unsqueeze(0), x2.unsqueeze(0)), dim=0)
        ynew = fcn(xnew) # (2, ..., q)
        msg = "Batched test fails (expanding batches changes the results)"
        assert torch.allclose(ynew[0], y), msg
        assert torch.allclose(ynew[1], y2), msg

    # generate shapes
    mv_xshapes = [
        (q,),
        (1,q),
        (1,1,q),
        (*batchshape, q),
        (1, *batchshape, q),
    ]
    mv_yshapes = [
        (*batchshape, p),
        (*batchshape, p) if len(batchshape) >= 1 else (1, p),
        (*batchshape, p) if len(batchshape) >= 2 else (1, 1, p),
        (*batchshape, p),
        (1, *batchshape, p)
    ]
    # test matvec and matmat, run input in multiple shapes to make sure no error is raised
    r = 2
    for (mv_xshape, mv_yshape) in zip(mv_xshapes, mv_yshapes):
        runtest("mv", mv_xshape, mv_yshape)
        runtest("mm", (*mv_xshape, r), (*mv_yshape, r))

    if not linop.is_rmv_implemented:
        return

    rmv_xshapes = [
        (p,),
        (1,p),
        (1,1,p),
        (*batchshape, p),
        (1, *batchshape, p),
    ]
    rmv_yshapes = [
        (*batchshape, q),
        (*batchshape, q) if len(batchshape) >= 1 else (1, q),
        (*batchshape, q) if len(batchshape) >= 2 else (1, 1, q),
        (*batchshape, q),
        (1, *batchshape, q)
    ]
    for (rmv_xshape, rmv_yshape) in zip(rmv_xshapes, rmv_yshapes):
        runtest("rmv", rmv_xshape, rmv_yshape)
        runtest("rmm", (*rmv_xshape, r), (*rmv_yshape, r))
