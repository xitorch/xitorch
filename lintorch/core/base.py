from abc import abstractmethod, abstractproperty
import torch

class BaseLinearModule(object):
    def __init__(self, shape,
                 is_symmetric=True, is_real=True):
        self._shape = shape
        self._is_symmetric = is_symmetric
        self._is_real = is_real

        # functions
        self._fcn_forward = None
        self._fcn_transpose = None
        self._fcn_precond = None

    ##################### functions #####################
    def __call__(self, *params, **kwargs):
        return self.forward(*params, **kwargs)

    def forward(self, *params, **kwargs):
        if self.is_forward_set():
            return self._fcn_forward(*params, **kwargs)
        raise RuntimeError("The forward function has not been defined.")

    def transpose(self, *params, **kwargs):
        if self.is_transpose_set():
            return self._fcn_transpose(*params, **kwargs)
        raise RuntimeError("The transpose function has not been defined.")

    def precond(self, *params, **kwargs):
        if self.is_precond_set():
            return self._fcn_precond(*params, **kwargs)
        raise RuntimeError("The preconditioning function has not been defined.")

    ##################### checkers #####################
    def is_forward_set(self):
        return self._fcn_forward is not None

    def is_transpose_set(self):
        return self._fcn_transpose is not None

    def is_precond_set(self):
        return self._fcn_precond is not None

    ##################### setters #####################
    def set_forward(self, fcn):
        # check arguments and check if _fcn_forward is defined ???
        self._fcn_forward = fcn
        if self.is_symmetric:
            self._fcn_transpose = fcn

    def set_transpose(self, fcn):
        # check arguments and check if _fcn_transpose is defined ???
        self._fcn_transpose = fcn

    def set_precond(self, fcn):
        # check arguments and check if _fcn_precond is defined ???
        self._fcn_precond = fcn

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

#################################### decor ####################################
def module(shape,
           is_symmetric=True,
           is_real=True):

    def decor(fcn):
        # check if it is a function (???)
        cls_module = BaseLinearModule(shape, is_symmetric, is_real)
        cls_module.set_forward(fcn)
        return cls_module

    return decor
