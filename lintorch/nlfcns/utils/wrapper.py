import inspect
from contextlib import contextmanager
import traceback as tb
import torch
from lintorch.core.editable_module import EditableModule
from lintorch.utils.attr import get_attr, set_attr, del_attr

def wrap_fcn(fcn, params):
    """
    Wrap function to include the object's parameters as well
    """


    # if the fcn is an object that has __call__ attribute, then assign it to fcn
    # to make fcn a method
    if not inspect.ismethod(fcn) and not inspect.isfunction(fcn):
        if hasattr(fcn, "__call__"):
            fcn = fcn.__call__
        else:
            raise RuntimeError("The function must be callable")

    # unroller object to make sure the parameters of the wrapped functions are
    # all tensors
    unroller = ParamsUnroller(params)
    nparams = unroller.num_all_tensors

    # if it is a method from an object, unroll the parameters and add
    # the object's parameters as well
    if inspect.ismethod(fcn) and isinstance(fcn.__self__, EditableModule):
        obj = fcn.__self__
        method_name = fcn.__name__
        def wrapped_fcn(*all_params):
            params = unroller.roll(all_params[:nparams])
            obj_params = all_params[nparams:]
            with obj.useparams(method_name, *obj_params) as model:
                res = fcn(*params)
            return res

        # get all the parameters
        obj_params = obj.getuniqueparams(method_name)
        all_params = [*unroller.unroll(params), *obj_params]

    # do the similar thing as EditableModule for torch.nn.Module, but using
    # obj.parameters() as the substitute as .getparams()
    elif inspect.ismethod(fcn) and isinstance(fcn.__self__, torch.nn.Module):
        obj = fcn.__self__

        # get the tensors in the torch.nn.Module to be used as params
        paramnames, obj_params = zip(*obj.named_parameters())
        all_params = [*unroller.unroll(params), *obj_params]

        def wrapped_fcn(*all_params2):
            params = unroller.roll(all_params2[:nparams])
            obj_params2 = all_params2[nparams:]
            # substitute obj.parameters() with obj_params
            with NNModuleUseParams(obj, paramnames, obj_params2):
                res = fcn(*params)
            return res

    # return as it is if fcn is just a function and params all are tensors
    elif unroller.all_tensors:
        wrapped_fcn = fcn
        all_params = params

    # if fcn is just a function and params are not all tensors (there is a
    # list of tensors), then unroll the parameters
    else:
        def wrapped_fcn(*all_params):
            params = unroller.roll(all_params)
            return fcn(params)
        all_params = unroller.unroll(params)

    return wrapped_fcn, all_params

class ParamsUnroller(object):
    """
    If one or more elements in params are list, then this class can provide
    functions to unroll it to a list of tensors or roll it back to the original
    form of params
    """
    def __init__(self, params):
        # format: 0 for tensor, 1 for list of tensors
        self.formats = [self._get_format(p) for p in params]
        self.lengths = [1 if f==0 else len(p) for p,f in zip(params, self.formats)]
        self._num_all_tensors = sum(self.lengths)
        self.params = params
        self.nparams = len(params)
        self._all_tensors = (sum(self.formats) == 0)

    @property
    def all_tensors(self):
        return self._all_tensors

    @property
    def num_all_tensors(self):
        return self._num_all_tensors

    def unroll(self, params):
        if self._all_tensors: return params
        res = []
        for i,p in enumerate(params):
            if self.formats[i] == 0:
                res.append(p)
            elif self.formats[i] == 1:
                res = res + p
        return res

    def roll(self, uparams):
        if self._all_tensors: return uparams
        res = []
        idx = 0
        for i in range(self.nparams):
            length = self.lengths[i]
            format = self.formats[i]
            if format == 0:
                res.append(uparams[idx])
            elif format == 1:
                res.append(uparams[idx:idx+length])
            idx += length
        return res

    def _get_format(self, p):
        if isinstance(p, torch.Tensor):
            return 0
        elif isinstance(p, list) or isinstance(p, tuple):
            return 1
        else:
            raise RuntimeError("Unknown type %s" % type(p))

@contextmanager
def NNModuleUseParams(nnmodule, names, params):
    try:
        # substitute the state dictionary of the module with the new tensor

        # save the current state
        state_tensors = [get_attr(nnmodule, name) for name in names]

        # substitute the state with the given tensor
        for (name, param) in zip(names, params):
            del_attr(nnmodule, name) # delete require in case the param is not a torch.nn.Parameter
            set_attr(nnmodule, name, param)

        yield nnmodule

    except Exception as exc:
        tb.print_exc()
    finally:
        # restore back the saved tensors
        for (name, param) in zip(names, state_tensors):
            set_attr(nnmodule, name, param)
