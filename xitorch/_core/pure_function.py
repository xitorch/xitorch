import torch
import inspect
from typing import Callable, Any, Optional
from xitorch._utils.attr import get_attr, set_attr, del_attr
from xitorch._core.editable_module import EditableModule
from contextlib import contextmanager
from abc import abstractmethod

__all__ = ["get_pure_function", "make_sibling"]

############################ functional ###############################
class PureFunction(object):
    """
    PureFunction class wraps methods to make it stateless and expose the pure
    function to take inputs of the original inputs (`params`) and the object's
    states (`objparams`).
    For functions, this class only acts as a thin wrapper.
    """
    def __init__(self, fcn, fcntocall=None):
        self._fcn = fcn
        self._fcntocall = fcn if fcntocall is None else fcntocall
        self._objparams = self.getobjparams()
        self._nobjparams = len(self._objparams)
        self._state_change_allowed = True

    def objparams(self):
        return self._objparams

    def allparams(self, params):
        return [*params, *self._objparams]

    def __call__(self, *params):
        return self._fcntocall(*params)

    @property
    def fcn(self):
        return self._fcn

    @abstractmethod
    def getobjparams(self):
        pass

    @contextmanager
    @abstractmethod
    def _useobjparams(self, objparams):
        pass

    @contextmanager
    def useobjparams(self, objparams):
        if not self._state_change_allowed:
            raise RuntimeError("The state change is disabled")
        with self._useobjparams(objparams):
            try: yield
            finally: pass

    @contextmanager
    def disable_state_change(self):
        try:
            prev_status = self._state_change_allowed
            self._state_change_allowed = False
            yield
        finally:
            self._state_change_allowed = prev_status

    # NOTE: please refrain from implementing enable_state_change.
    # If you want to enable state change, then just remove
    # `with obj.disable_state_change()` statement in your code

class EditableModulePureFunction(PureFunction):
    def getobjparams(self):
        fcn = self.fcn
        self.obj = fcn.__self__
        self.methodname = fcn.__name__
        objparams = self.obj.getuniqueparams(self.methodname)
        return objparams

    @contextmanager
    def _useobjparams(self, objparams):
        with self.obj.useuniqueparams(self.methodname, *objparams):
            try: yield
            finally: pass

class TorchNNPureFunction(PureFunction):
    def getobjparams(self):
        fcn = self.fcn
        obj = fcn.__self__

        # get the tensors in the torch.nn.Module to be used as params
        named_params = list(obj.named_parameters())
        if len(named_params) == 0:
            paramnames = []
            obj_params = []
        else:
            paramnames, obj_params = zip(*named_params)
        self.names = paramnames
        self.obj = obj
        return list(obj_params)

    @contextmanager
    def _useobjparams(self, objparams):
        nnmodule = self.obj
        names = self.names
        try:
            # substitute the state dictionary of the module with the new tensor

            # save the current state
            state_tensors = [get_attr(nnmodule, name) for name in names]

            # substitute the state with the given tensor
            for (name, param) in zip(names, objparams):
                del_attr(nnmodule, name) # delete require in case the param is not a torch.nn.Parameter
                set_attr(nnmodule, name, param)

            yield nnmodule

        except Exception as exc:
            raise exc
        finally:
            # restore back the saved tensors
            for (name, param) in zip(names, state_tensors):
                set_attr(nnmodule, name, param)

class FunctionPureFunction(PureFunction):
    def getobjparams(self):
        return []

    @contextmanager
    def _useobjparams(self, objparams):
        try: yield
        finally: pass

def make_sibling(pfunc:Callable[...,Any]) -> Callable[[Callable[...,Any]], PureFunction]:
    """
    Used as a decor to mark the decorated function as a sibling method of the
    input ``pfunc``.
    Sibling method is a method that is virtually belong to the same object, but
    behaves differently.
    Changing the state of the decorated function will also change the state of
    ``pfunc`` and its other siblings.
    """
    if not isinstance(pfunc, PureFunction):
        pfunc = _get_pure_function(pfunc)
    def decor(fcn):
        new_pfunc = _get_pure_function(pfunc.fcn, fcntocall=fcn)
        return new_pfunc
    return decor

def get_pure_function(fcn:Callable[...,Any]) -> PureFunction:
    """
    Get the pure function form of the function or method ``fcn``.

    Arguments
    ---------
    fcn: function or method
        Function or method to be converted into a ``PureFunction`` by exposing
        the hidden parameters affecting its outputs.

    Returns
    -------
    PureFunction
        The pure function wrapper
    """
    return _get_pure_function(fcn)

def _get_pure_function(
        fcn,
        fcntocall:Optional[Callable[...,Any]]=None) -> PureFunction:
    if not inspect.ismethod(fcn) and not inspect.isfunction(fcn) and not isinstance(fcn, PureFunction):
        if hasattr(fcn, "__call__"):
            fcn = fcn.__call__
        else:
            raise RuntimeError("The function must be callable")

    if isinstance(fcn, PureFunction):
        pfunc = fcn

    elif inspect.isfunction(fcn):
        pfunc = FunctionPureFunction(fcn, fcntocall=fcntocall)

    # if it is a method from an object, unroll the parameters and add
    # the object's parameters as well
    elif isinstance(fcn.__self__, EditableModule):
        pfunc = EditableModulePureFunction(fcn, fcntocall=fcntocall)

    # do the similar thing as EditableModule for torch.nn.Module, but using
    # obj.parameters() as the substitute as .getparams()
    elif isinstance(fcn.__self__, torch.nn.Module):
        pfunc = TorchNNPureFunction(fcn, fcntocall=fcntocall)

    # return as it is if fcn is just a function and params all are tensors
    else:
        raise RuntimeError("The fcn must be a Python function or a method of torch.nn.Module or xitorch.EditableModule")

    return pfunc
