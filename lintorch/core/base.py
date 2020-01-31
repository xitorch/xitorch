from abc import abstractmethod, abstractproperty
import torch

class BaseLinearModule(torch.nn.Module):
    """
    Base module of linear modules.
    Linear module is a module that can be expressed as a matrix of size
        (nrows, ncols).
    """
    ################# functions / properties to be implemented #################
    @abstractmethod
    def _forward(self, x, params):
        """
        Calculate the operation of the transformation with `x` where
        the detail of the transformation is set by params.
        `x` and each of `params` should be differentiable.

        Arguments
        ---------
        * x: torch.tensor (nbatch, ncols) or (nbatch, ncols, nelmt)
            The input vector of the linear transformation.
        * params: list of torch.tensor (nbatch, ...)
            The differentiable parameters that sets the linear transformation.

        Returns
        -------
        * y: torch.tensor (nbatch, nrows) or (nbatch, nrows, nelmt)
            The output of the linear transformation, i.e. y = Ax
        """
        pass

    def _transpose(self, y, params):
        """
        Calculate the operation of the transpose transformation with `y` where
        the detail of the transformation is set by params.
        `y` and each of `params` should be differentiable.

        Arguments
        ---------
        * y: torch.tensor (nbatch, nrows, nelmt)
            The input vector of the linear transformation.
        * params: list of torch.tensor (nbatch, ...)
            The differentiable parameters that sets the linear transformation.

        Returns
        -------
        * x: torch.tensor (nbatch, ncols, nelmt)
            The output of the linear transformation, i.e. x = A.T y
        """
        msg = "The ._transpose() method is unimplemented for class %s" % \
              (self.__class__.__name__)
        raise RuntimeError(msg)

    def _precond(self, y, params):
        """
        Apply the preconditioning matrix of this transformation.
        Preconditioning is an approximate of the inverse of the matrix but
        it should be much cheaper to calculate.

        Arguments
        ---------
        * y: torch.tensor (nbatch, nrows, nelmt)
            The input vector of the linear transformation.
        * params: list of torch.tensor (nbatch, ...)
            The differentiable parameters that sets the linear transformation.

        Returns
        -------
        * x: torch.tensor (nbatch, ncols, nelmt)
            The output of the linear transformation, i.e. x = M^(-1) y
        """
        msg = "The .precond() method is unimplemented for class %s" % \
              (self.__class__.__name__)
        raise RuntimeError(msg)

    @abstractproperty
    def shape(self):
        """
        Returns (nrows, ncols)
        """
        pass

    @abstractproperty
    def issymmetric(self):
        pass

    @abstractproperty
    def nparams(self):
        pass

    ##################### implemented properties #####################
    @property
    def issquare(self):
        return self.shape[0] == self.shape[1]

    ##################### arithmetic operator #####################
    def __add__(self, other):
        other = _normalize_linear_module(self, other)
        return AddModule(self, other)

    def __sub__(self, other):
        other = _normalize_linear_module(self, other)
        return AddModule(self, -other)

    def __neg__(self):
        return NegModule(self)

    ##################### functions to be called #####################
    def __call__(self, x, params):
        return self.forward(x, params)

    def forward(self, x, params):
        """
        Calculate the operation of the transformation with `x` where
        the detail of the transformation is set by params.
        `x` and each of `params` should be differentiable.

        Arguments
        ---------
        * x: torch.tensor (nbatch, ncols) or (nbatch, ncols, nelmt)
            The input vector of the linear transformation.
        * params: list of torch.tensor (nbatch, ...)
            The differentiable parameters that sets the linear transformation.

        Returns
        -------
        * y: torch.tensor (nbatch, nrows) or (nbatch, nrows, nelmt)
            The output of the linear transformation, i.e. y = Ax
        """
        xndim = x.ndim
        if xndim == 2:
            x = x.unsqueeze(-1)
        y = self._forward(x, params)
        if xndim == 2:
            y = y.squeeze(-1)
        return y

    def transpose(self, y, params):
        """
        Calculate the operation of the transpose transformation with `y` where
        the detail of the transformation is set by params.
        `y` and each of `params` should be differentiable.

        Arguments
        ---------
        * y: torch.tensor (nbatch, nrows) or (nbatch, nrows, nelmt)
            The input vector of the linear transformation.
        * params: list of torch.tensor (nbatch, ...)
            The differentiable parameters that sets the linear transformation.

        Returns
        -------
        * x: torch.tensor (nbatch, ncols) or (nbatch, ncols, nelmt)
            The output of the linear transformation, i.e. x = A.T y
        """
        yndim = y.ndim
        if yndim == 2:
            y = y.unsqueeze(-1)

        if self.issymmetric:
            x = self._forward(y, params)
        else:
            x = self._transpose(y, params)

        if yndim == 2:
            x = x.squeeze(-1)
        return x

    def precond(self, y, params):
        if hasattr(self, "_precond_") and self._precond_ is not None:
            return self._precond_(y, params)
        else:
            return self._precond(y, params)

    def setprecond(self, pcond):
        self._precond_ = pcond

    @property
    def T(self):
        """
        Returns the transposed object.
        """
        if not hasattr(self, "_T_"):
            self._T_ = TransposeModule(self)
        return self._T_

    def getfunc(self, params):
        # returns the forward function as a callable
        def A(x, params):
            return self.forward(x, params)
        return A

def _normalize_linear_module(self, a):
    if isinstance(a, BaseLinearModule):
        return a
    raise RuntimeError("Unknown type")
    # elif type(a) in [int, float]:
    #     return a # ???

class TransposeModule(BaseLinearModule):
    def __init__(self, model):
        super(TransposeModule, self).__init__()
        self.model = model
        self._shape = [model.shape[1], model.shape[0]]

    def _forward(self, x, params):
        return self.model._transpose(x, params)

    def _transpose(self, x, params):
        return self.model._forward(x, params)

    def _precond(self, y, params):
        return self.model._precond(y, params)

    @property
    def T(self):
        return self.model

    @property
    def nparams(self):
        return self.model.nparams

    @property
    def shape(self):
        return self._shape

    @property
    def issymmetric(self):
        return self.model.issymmetric

class AddModule(BaseLinearModule):
    def __init__(self, a, b):
        super(AddModule, self).__init__()
        self.a = a
        self.b = b
        self._nparams = self.a.nparams + self.b.nparams

    def _forward(self, x, params):
        aparams, bparams = self._split_params(params)
        return self.a._forward(x, aparams) + self.b._forward(x, bparams)

    def _transpose(self, y, params):
        aparams, bparams = self._split_params(params)
        return self.a._transpose(y, aparams) + self.b._transpose(y, bparams)

    @property
    def shape(self):
        return self.a.shape

    @property
    def issymmetric(self):
        # NOTE: this might be wrong!
        return self.a.issymmetric and self.b.issymmetric

    @property
    def nparams(self):
        return self._nparams

    def _split_params(self, params):
        aparams = params[:self.a.nparams]
        bparams = params[self.a.nparams:]
        return aparams, bparams

class NegModule(BaseLinearModule):
    def __init__(self, a):
        super(NegModule, self).__init__()
        self.a = a
        self._nparams = self.a.nparams

    def _forward(self, x, params):
        return -self.a._forward(x, params)

    def _transpose(self, y, params):
        return -self.a._transpose(y, params)

    def _precond(self, y, params):
        return -self.a._precond(y, params)

    @property
    def shape(self):
        return self.a.shape

    @property
    def issymmetric(self):
        return self.a.issymmetric

    @property
    def nparams(self):
        return self.a.nparams

class EyeModule(BaseLinearModule):
    def __init__(self, shape, val=1.0):
        super(EyeModule, self).__init__()
        self.val = val
        self._shape = shape

    def _forward(self, x, params):
        return self.val * x

    @property
    def issymmetric(self):
        return True

    @property
    def shape(self):
        return self._shape

    @property
