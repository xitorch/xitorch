import inspect
from contextlib import contextmanager
import traceback as tb
import torch
from comptorch.core.module import Module
from comptorch.utils.attr import get_attr, set_attr, del_attr

def get_wrap_fcn(fcn, params):
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

    nparams = len(params)

    # if it is a method from a module object, add the object's parameters as
    # the new funtion's methods
    if inspect.ismethod(fcn) and \
            (isinstance(fcn.__self__, Module) or isinstance(fcn.__self__, torch.nn.Module)):
        obj = fcn.__self__

        # get the tensors in the torch.nn.Module to be used as params
        paramnames, obj_params = zip(*obj.named_parameters())
        all_params = [*params, *obj_params]

        def wrapped_fcn(*all_params2):
            params = all_params2[:nparams]
            obj_params2 = all_params2[nparams:]
            # substitute obj.parameters() with obj_params
            with useparams(obj, paramnames, obj_params2):
                res = fcn(*params)
            return res

    # return as it is if fcn is just a function and params all are tensors
    else:
        wrapped_fcn = fcn
        all_params = params

    return wrapped_fcn, all_params

@contextmanager
def useparams(module, names, params):
    try:
        # substitute the state dictionary of the module with the new tensor

        # save the current state
        state_tensors = [get_attr(module, name) for name in names]

        # substitute the state with the given tensor
        for (name, param) in zip(names, params):
            del_attr(module, name) # delete require in case the param is not a torch.nn.Parameter
            set_attr(module, name, param)

        yield module

    except Exception as exc:
        tb.print_exc()
    finally:
        # restore back the saved tensors
        for (name, param) in zip(names, state_tensors):
            set_attr(module, name, param)
