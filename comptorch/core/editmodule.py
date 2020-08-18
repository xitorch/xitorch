import inspect
from contextlib import contextmanager
import traceback as tb
import torch
from comptorch.core.module import CModule, _param_traverser
from comptorch.utils.attr import get_attr, set_attr, del_attr

def get_wrap_fcn(fcn, params):
    """
    Wrap function to include the object's parameters as well.

    Arguments
    ---------
    * fcn: callable
        A standalone callable
    """
    if not hasattr(fcn, "__call__"):
        raise RuntimeError("The fcn argument must be callable.")

    # if it is a function, then return as it is (it has no object parameters)
    if inspect.isfunction(fcn):
        return fcn, params

    # if the fcn is an object that has __call__ attribute, then assign it to fcn
    # to make fcn a method
    if not inspect.ismethod(fcn):
        fcn = fcn.__call__

    obj = fcn.__self__
    if not isinstance(obj, CModule) and not isinstance(obj, torch.nn.Module):
        raise RuntimeError("The object of the method must be comptorch.CModule or torch.nn.Module")

    nparams = len(params)

    # if it is a method from a module object, add the object's parameters as
    # the new funtion's methods

    # get the tensors in the torch.nn.Module to be used as params
    named_params = dict(obj.named_parameters(recurse=True))
    paramnames = list(named_params.keys())
    obj_params = list(named_params.values())
    all_params = [*params, *obj_params]

    unique_map = _get_uniqueness_map(obj, paramnames)
    def wrapped_fcn(*all_params2):
        params = all_params2[:nparams]
        obj_params2 = all_params2[nparams:]
        # substitute obj.parameters() with obj_params
        with _useparams(obj, paramnames, obj_params2, unique_map):
            res = fcn(*params)
        return res

    return wrapped_fcn, all_params

def _get_uniqueness_map(module, names):
    # returns a mapping from the element of `names` into list of names that share
    # the same tensor (the value include itself)
    # the tensor corresponding to names are guaranteed (by user) to be unique.
    # the unique name can be obtained by calling .named_parameters()
    params = []
    idparam2name = {}
    res = {}
    for name in names:
        p = get_attr(module, name)
        params.append(p)
        idparam2name[id(p)] = name
        res[name] = []

    # traverses down the module to find the same tensor
    for name, v in _param_traverser(module):
        idv = id(v)
        if idv in idparam2name:
            unique_name = idparam2name[idv]
            res[unique_name].append(name)
    return res

@contextmanager
def _useparams(module, names, params, unique_map):
    try:
        # substitute the state dictionary of the module with the new tensor

        # save the current state
        state_tensors = [get_attr(module, name) for name in names]

        # substitute the state with the given tensor
        for (name, new_param, old_param) in zip(names, params, state_tensors):
            if old_param is new_param:
                continue
            for n in unique_map[name]:
                del_attr(module, n) # delete require in case the new_param is not a nn.Parameter and old_param is a nn.Parameter
                set_attr(module, n, new_param)

        yield module

    except Exception as exc:
        raise exc
        # tb.print_exc()
    finally:
        # restore back the saved tensors
        for (name, new_param, old_param) in zip(names, params, state_tensors):
            if old_param is new_param:
                continue
            for n in unique_map[name]:
                del_attr(module, n)
                set_attr(module, n, old_param)
