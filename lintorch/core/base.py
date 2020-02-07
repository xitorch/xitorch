import inspect
from abc import abstractmethod, abstractproperty
import torch
from lintorch.utils.exceptions import UnimplementedError

__all__ = ["Module", "module", "module_like"]

class Module(torch.nn.Module):
    def __init__(self, shape,
               is_symmetric=True,
               is_real=True):
        super(Module, self).__init__()

        self._shape = shape
        self._is_symmetric = is_symmetric
        self._is_real = is_real

        self._fcn_forward = None
        self._fcn_transpose = None
        self._fcn_precond = None
        self._is_forward_set = False
        self._is_transpose_set = False
        self._is_precond_set = False

        # optional arguments
        self._precond_opt_args = {
            "biases": None,
            "M": None,
            "mparams": []
        }

        # if the class is inherited, then check the implemented method
        self._inherited = self.__class__ != Module
        if self._inherited:
            # check the methods available in the class
            self._is_forward_set = self._check_fcn("forward")
            if self._is_symmetric and self._is_forward_set:
                self._is_transpose_set = True
                self._fcn_transpose = self.forward
            else:
                self._is_transpose_set = self._check_fcn("transpose")
            self._is_precond_set = self._check_fcn("precond")

    def _check_fcn(self, fcnname):
        fcn = getattr(self, fcnname)
        x = torch.zeros(1,self.shape[1],1)
        try:
            fcn(x)
        except UnimplementedError:
            return False
        except:
            return True
        return True

    def __call__(self, x, *params):
        """
        Apply the transformation to x.

        Arguments
        ---------
        * x: torch.tensor (nbatch, nc, ncols)
            The tensor to be applied the transformation.
        * *params: list of torch.tensor (nbatch, ...)
            List of torch tensor that set the transformation.

        Returns
        -------
        * y: torch.tensor (nbatch, nr, ncols)
            The tensor of the transformation result.
        """
        return self.forward(x, *params)

    def forward(self, x, *params):
        """
        Apply the transformation to x.

        Arguments
        ---------
        * x: torch.tensor (nbatch, nc, ncols)
            The tensor to be applied the transformation.
        * *params: list of torch.tensor (nbatch, ...)
            List of torch tensor that set the transformation.

        Returns
        -------
        * y: torch.tensor (nbatch, nr, ncols)
            The tensor of the transformation result.
        """
        if self.is_forward_set():
            return self._fcn_forward(x, *params)
        raise UnimplementedError("The transpose function has not been defined.")

    def transpose(self, x, *params):
        """
        Apply the transpose transformation to x.

        Arguments
        ---------
        * x: torch.tensor (nbatch, nr, ncols)
            The tensor to be applied the transpose transformation.
        * *params: list of torch.tensor (nbatch, ...)
            List of torch tensor that set the transformation.

        Returns
        -------
        * y: torch.tensor (nbatch, nc, ncols)
            The tensor of the transpose transformation result.
        """
        if self.is_transpose_set():
            return self._fcn_transpose(x, *params)
        raise UnimplementedError("The transpose function has not been defined.")

    def precond(self, x, *params, biases=None, M=None, mparams=[]):
        """
        Approximate the solution of Ay=x or (A-biases*I)y=x.

        Arguments
        ---------
        * x: torch.tensor (nbatch, nr, ncols)
            The tensor to be applied the inverse transformation.
        * *params: list of torch.tensor (nbatch, ...)
            List of torch tensor that set the transformation.
        * biases: torch.tensor (nbatch, ncols) or None
            If None, then it solves Ay=x. Otherwise, it solves (A-biases*I)y=x
            for different biases for every columns.
        * M: lintorch.Module or None
            The transformation on the biases side. If biases is None,
            then this argument is ignored. If None or ignored, then M=I.
        * mparams: list of differentiable torch.tensor
            List of differentiable torch.tensor to be put to M.

        Returns
        -------
        * y: torch.tensor (nbatch, nc, ncols)
            The tensor of the inverse result.
        """
        if self.is_precond_set():
            return self._fcn_precond(x, *params, biases=biases, M=M, mparams=mparams)
        raise UnimplementedError("The preconditioning function has not been defined.")

    ##################### checkers #####################
    def is_forward_set(self):
        return self._is_forward_set

    def is_transpose_set(self):
        return self._is_transpose_set

    def is_precond_set(self):
        return self._is_precond_set

    ##################### setters #####################
    def set_forward(self, fcn):
        # check arguments and check if _fcn_precond is defined ???
        self._fcn_forward = fcn
        self._is_forward_set = True
        if self.is_symmetric:
            self.set_transpose(fcn)
        return fcn

    def set_transpose(self, fcn):
        # check arguments and check if _fcn_transpose is defined ???
        self._fcn_transpose = fcn
        self._is_transpose_set = True
        return fcn

    def set_precond(self, fcn):
        # check arguments and check if _fcn_precond is defined ???
        self._fcn_precond = fcn
        self._is_precond_set = True
        return fcn

    ##################### properties #####################
    @property
    def shape(self):
        return self._shape

    @property
    def is_symmetric(self):
        return self._is_symmetric

    @property
    def is_real(self):
        return self._is_real

    ##################### checkers #####################
    def is_forward_set(self):
        return self._is_forward_set

    def is_transpose_set(self):
        return self._is_transpose_set

    def is_precond_set(self):
        return self._is_precond_set

    ##################### implemented functions #####################
    def fullmatrix(self, *params):
        """
        Returns the full matrix of the module.
        Warning: if your matrix is too big, then calling this function will
        drain your memory.
        """

        nbatch = params[0].shape[0]
        na = self.shape[0]
        dtype, device = self._get_dtype_device(params)
        V = torch.eye(na).unsqueeze(0).expand(nbatch,-1,-1).to(dtype).to(device)

        # obtain the full matrix of A
        return self.forward(V, *params)

    ##################### private functions #####################
    def _get_dtype_device(params):
        A_params = list(self.parameters())
        if len(A_params) == 0:
            p = params[0]
        else:
            p = A_params[0]
        dtype = p.dtype
        device = p.device
        return dtype, device


#################################### decor ####################################
def module(shape,
           is_symmetric=True,
           is_real=True):

    def decor(fcn):
        # check if it is a function (???)
        cls_module = Module(shape, is_symmetric, is_real)
        cls_module.set_forward(fcn)
        return cls_module

    return decor

def module_like(A):
    return module(
        shape = A.shape,
        is_symmetric = A.is_symmetric,
        is_real = A.is_real
    )

if __name__ == "__main__":
    na = 25

    @module(shape=(na,na))
    def A(x, diag):
        return x * diag

    @A.set_precond
    def precond(x, diag, biases=None):
        return x / diag

    class B(Module):
        def __init__(self):
            super(B, self).__init__(shape=(na, na))

        def forward(self, x, diag):
            return x * diag

        def precond(self, y, diag, biases=None):
            return y / diag

    dtype = torch.float64
    x = torch.ones(1,na,1).to(dtype)
    diag = (torch.arange(na)+1.0).unsqueeze(0).unsqueeze(-1).to(dtype)
    y = A(x, diag)
    x0 = A.precond(y, diag)
    print(A.transpose(y, diag).squeeze())
    print(y.squeeze())
    print(x0.squeeze())
    b = B()
    by = b(x, diag)
    print(by.squeeze())
    print(b.precond(by, diag).squeeze())
