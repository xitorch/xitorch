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

    nparams = len(params)
    if inspect.isfunction(fcn):
        return fcn, params

    # if it is a method from an object, unroll the parameters and add
    # the object's parameters as well
    if isinstance(fcn.__self__, EditableModule):
        obj = fcn.__self__
        method_name = fcn.__name__
        def wrapped_fcn(*all_params):
            params = all_params[:nparams]
            obj_params = all_params[nparams:]
            with obj.useparams(method_name, *obj_params) as model:
                res = fcn(*params)
            return res

        # get all the parameters
        obj_params = obj.getuniqueparams(method_name)
        all_params = [*params, *obj_params]

    # do the similar thing as EditableModule for torch.nn.Module, but using
    # obj.parameters() as the substitute as .getparams()
    elif isinstance(fcn.__self__, torch.nn.Module):
        obj = fcn.__self__

        # get the tensors in the torch.nn.Module to be used as params
        paramnames, obj_params = zip(*obj.named_parameters())
        all_params = [*params, *obj_params]

        def wrapped_fcn(*all_params2):
            params = all_params2[:nparams]
            obj_params2 = all_params2[nparams:]
            # substitute obj.parameters() with obj_params
            with NNModuleUseParams(obj, paramnames, obj_params2):
                res = fcn(*params)
            return res

    # return as it is if fcn is just a function and params all are tensors
    else:
        raise RuntimeError("The fcn must be a method of torch.nn.Module or lintorch.EditableModule")

    return wrapped_fcn, all_params

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
