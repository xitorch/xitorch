import torch
import inspect
from lintorch._utils.attr import get_attr, set_attr, del_attr
from lintorch._core.editable_module import EditableModule
from contextlib import contextmanager
from abc import abstractmethod

__all__ = ["wrap_fcn"]

############################ functional ###############################
class PureFunction(object):
    """
    PureFunction class wraps methods to make it stateless and expose the pure
    function to take inputs of the original inputs (`params`) and the object's
    states (`objparams`).
    For functions, this class only acts as a thin wrapper.
    """
    def __init__(self, fcn, params):
        self._nparams = len(params)
        self._fcn = fcn
        self._objparams = self.getobjparams(fcn)
        self._params = params

    def allparams(self):
        return [*self._params, *self._objparams]

    def __call__(self, *all_params, sameobj=False):
        params = all_params[:self._nparams]
        if sameobj:
            return self._fcn(*params)
        else:
            objparams = all_params[self._nparams:]
            with self.useobjparams(objparams):
                return self._fcn(*params)

    @property
    def fcn(self):
        return self._fcn

    @abstractmethod
    def getobjparams(self, fcn):
        pass

    @contextmanager
    @abstractmethod
    def useobjparams(self, objparams):
        pass

class EditableModulePureFunction(PureFunction):
    def getobjparams(self, fcn):
        self.obj = fcn.__self__
        self.methodname = fcn.__name__
        objparams = self.obj.getuniqueparams(self.methodname)
        return objparams

    @contextmanager
    def useobjparams(self, objparams):
        with self.obj.useuniqueparams(self.methodname, *objparams):
            try:
                yield
            finally:
                pass

class TorchNNPureFunction(PureFunction):
    def getobjparams(self, fcn):
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
        return obj_params

    @contextmanager
    def useobjparams(self, objparams):
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
    def getobjparams(self, fcn):
        return []

    @contextmanager
    def useobjparams(self, objparams):
        try: yield
        finally: pass

def wrap_fcn(fcn, params):
    """
    Wrap function to include the object's parameters as well
    """

    # if the fcn is an object that has __call__ attribute, then assign it to fcn
    # to make fcn a method
    if isinstance(params, torch.Tensor):
        params = [params]

    if not inspect.ismethod(fcn) and not inspect.isfunction(fcn) and not isinstance(fcn, PureFunction):
        if hasattr(fcn, "__call__"):
            fcn = fcn.__call__
        else:
            raise RuntimeError("The function must be callable")

    if isinstance(fcn, PureFunction):
        pfunc = FunctionPureFunction(fcn.__call__, params)

    elif inspect.isfunction(fcn):
        pfunc = FunctionPureFunction(fcn, params)

    # if it is a method from an object, unroll the parameters and add
    # the object's parameters as well
    elif isinstance(fcn.__self__, EditableModule):
        pfunc = EditableModulePureFunction(fcn, params)

    # do the similar thing as EditableModule for torch.nn.Module, but using
    # obj.parameters() as the substitute as .getparams()
    elif isinstance(fcn.__self__, torch.nn.Module):
        pfunc = TorchNNPureFunction(fcn, params)

    # return as it is if fcn is just a function and params all are tensors
    else:
        raise RuntimeError("The fcn must be a method of torch.nn.Module or lintorch.EditableModule")

    return pfunc, pfunc.allparams()
