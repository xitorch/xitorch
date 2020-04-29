import inspect
from abc import abstractmethod, abstractproperty
import torch
from lintorch.utils.exceptions import UnimplementedError
from lintorch.core.editable_module import EditableModule, getmethodparams, setmethodparams

__all__ = ["Module", "module", "module_like"]

class Module(EditableModule):
    def __init__(self, shape,
               is_symmetric=True,
               is_real=True,
               dtype=None,
               device=None):
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
        self._transposed_module = None

        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = torch.device("cpu")
        self._device = device
        self._dtype = dtype

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
                # if there is no transpose function defined by the user,

            self._is_precond_set = self._check_fcn("precond")

        # use the default transpose
        if not self._is_transpose_set:
            self._is_transpose_set = True
            self._fcn_transpose = self._default_transpose

    def to(self, dtype_or_device):
        # super(Module, self).to(dtype_or_device)
        if isinstance(dtype_or_device, torch.dtype):
            self._dtype = dtype_or_device
        elif isinstance(dtype_or_device, torch.device):
            self._device = dtype_or_device
        else:
            raise TypeError("The arguments of .to() can only be torch dtype or device.")
        return self

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
        raise UnimplementedError("The forward function has not been defined.")

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
        raise UnimplementedError("The transpose function has not been defined")

    def _default_transpose(self, x, *params):
        # the transpose is simply the backward propagation
        with torch.enable_grad():
            x1 = x.detach().requires_grad_()
            y1 = self.forward(x1, *params)
        res = torch.autograd.grad(y1, (x1,), grad_outputs=(x,),
            create_graph=torch.is_grad_enabled())[0]
        return res

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

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

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
        if len(params) == 0:
            try:
                return self.__fullmatrix_
            except AttributeError: pass

        nbatch = params[0].shape[0] if len(params) > 0 else 1
        na = self.shape[0]
        dtype = self._dtype
        device = self._device
        V = torch.eye(na).unsqueeze(0).expand(nbatch,-1,-1).to(dtype).to(device)

        # obtain the full matrix of A
        mat = self.forward(V, *params)
        # doing this could improve numerical stability
        if self.is_symmetric:
            mat = (mat + mat.transpose(-2,-1)) * 0.5

        # save the matrix if there is no params
        if len(params) == 0:
            self.__fullmatrix_ = mat
        return mat

    @property
    def T(self):
        if self._transposed_module is None:
            self._transposed_module = TransposeModule(self)
        return self._transposed_module

    ##################### editable module part #####################
    def getparams(self, methodname):
        # TODO: check if it is inherited or not
        if (methodname == "forward" or methodname == "__call__") and self.is_forward_set():
            return getmethodparams(self._fcn_forward)
        elif methodname == "transpose" and self.is_transpose_set():
            return getmethodparams(self._fcn_transpose)
        elif methodname == "fullmatrix":
            try:
                return [self.__fullmatrix_]
            except AttributeError:
                return self.getparams("forward")
        else:
            raise RuntimeError("The method %s is not defined for getparams" % methodname)

    def setparams(self, methodname, *params):
        if (methodname == "forward" or methodname == "__call__") and self.is_forward_set():
            return setmethodparams(self._fcn_forward, *params)
        elif methodname == "transpose" and self.is_transpose_set():
            return setmethodparams(self._fcn_transpose, *params)
        elif methodname == "fullmatrix":
            try:
                self.__fullmatrix_, = params[:1]
                return 1
            except AttributeError:
                return self.setparams("forward", *params)
        else:
            raise RuntimeError("The method %s is not defined for setparams" % methodname)

class TransposeModule(Module):
    def __init__(self, model):
        super(TransposeModule, self).__init__(
            shape=model.shape,
            is_symmetric=model.is_symmetric,
            is_real=model.is_real,
            dtype=model.dtype,
            device=model.device)
        self.model = model

    def to(self, dtype_or_device):
        self.model.to(dtype_or_device)

    def forward(self, x, *params):
        return self.model.transpose(x, *params)

    def transpose(self, x, *params):
        return self.model.forward(x, *params)

    def precond(self, x, *params, biases=None, M=None, mparams=[]):
        return self.model.precond(x, *params, biases=biases, M=M, mparams=mparams)

    ##################### checkers #####################
    def is_forward_set(self):
        return self.model.is_transpose_set()

    def is_transpose_set(self):
        return self.model.is_forward_set()

    def is_precond_set(self):
        return self.model.is_precond_set()

    ##################### setters #####################
    def set_forward(self, fcn):
        return self.model.set_transpose(fcn)

    def set_transpose(self, fcn):
        return self.model.set_forward(fcn)

    def set_precond(self, fcn):
        return self.model.set_precond(fcn)

    ##################### implemented functions #####################
    def fullmatrix(self, *params):
        return self.model.fullmatrix(*params).transpose(-2,-1)

    @property
    def T(self):
        return self.model

    ##################### editable module part #####################
    def getparams(self, methodname):
        return self.model.getparams(methodname)

    def setparams(self, methodname, *params):
        return self.model.setparams(methodname, *params)

#################################### decor ####################################
def module(shape,
           is_symmetric=True,
           is_real=True,
           dtype=None,
           device=None):

    def decor(fcn):
        # check if it is a function (???)
        cls_module = Module(shape, is_symmetric, is_real, dtype=dtype, device=device)
        cls_module.set_forward(fcn)
        return cls_module

    return decor

def module_like(A):
    return module(
        shape = A.shape,
        is_symmetric = A.is_symmetric,
        is_real = A.is_real,
        dtype = A.dtype,
        device = A.device
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
