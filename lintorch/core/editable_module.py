import inspect
from abc import abstractmethod
from contextlib import contextmanager
import copy
import torch

__all__ = ["EditableModule", "list_operating_params"]

class EditableModule(object):
    @abstractmethod
    def getparams(self, methodname):
        """
        Returns a list of tensor parameters used in the object's operations
        """
        pass

    @abstractmethod
    def setparams(self, methodname, *params):
        """
        Set the input parameters to the object's parameters to make a copy of
        the operations.
        """
        pass

    @contextmanager
    def useparams(self, methodname, *params):
        try:
            _orig_params_ = self.getparams(methodname)
            self.setparams(methodname, *params)
            yield self
        finally:
            self.setparams(methodname, *_orig_params_)

def getmethodparams(method):
    if not inspect.ismethod(method):
        return []
    obj = method.__self__
    methodname = method.__name__
    if not isinstance(obj, EditableModule):
        return []
    return obj.getparams(methodname)

def setmethodparams(method, *params):
    if not inspect.ismethod(method):
        return
    obj = method.__self__
    methodname = method.__name__
    if not isinstance(obj, EditableModule):
        return
    obj.setparams(methodname, *params)

############################ debugging functions ############################

def list_operating_params(method, *args, **kwargs):
    """
    List the tensors used in executing the method
    """
    obj = method.__self__

    # get all the tensors recursively
    max_depth = 3
    all_tensors, all_names = _get_tensors(obj, prefix="self", max_depth=max_depth)

    # copy the tensors and require them to be differentiable
    copy_tensors0 = [tensor.clone().detach().requires_grad_() for tensor in all_tensors]
    copy_tensors = copy.copy(copy_tensors0)
    _set_tensors(obj, copy_tensors, max_depth=max_depth)

    # run the method and see which one has the gradients
    output = method(*args, **kwargs).sum()
    grad_tensors = torch.autograd.grad(output, copy_tensors0, allow_unused=True)

    res = []
    for i, grad in enumerate(grad_tensors):
        if grad is None:
            continue
        res.append(all_names[i])

    # print the results
    res_str = ", ".join(res)
    print("'%s': [%s]," % (method.__name__, res_str))

def _get_tensors(obj, prefix, max_depth=4):
    # get the tensors recursively towards torch.nn.Module
    res = []
    names = []
    float_type = [torch.float32, torch.float, torch.float64, torch.float16]
    for key in obj.__dict__:
        elmt = obj.__dict__[key]
        name = "%s.%s"%(prefix, key)
        if isinstance(elmt, torch.Tensor) and elmt.dtype in float_type:
            res.append(elmt)
            names.append(name)
        elif hasattr(elmt, "__dict__"):
            new_res = []
            new_names = []
            if isinstance(elmt, torch.nn.Module):
                new_res, new_names = _get_tensors(elmt, prefix=name, max_depth=max_depth)
            elif max_depth > 0:
                new_res, new_names = _get_tensors(elmt, prefix=name, max_depth=max_depth-1)
            res = res + new_res
            names = names + new_names
    return res, names

def _set_tensors(obj, all_params, max_depth=4):
    float_type = [torch.float32, torch.float, torch.float64, torch.float16]
    for key in obj.__dict__:
        elmt = obj.__dict__[key]
        if isinstance(elmt, torch.Tensor) and elmt.dtype in float_type:
            obj.__dict__[key] = all_params.pop(0)
        elif hasattr(elmt, "__dict__"):
            if isinstance(elmt, torch.nn.Module):
                _set_tensors(elmt, all_params, max_depth=max_depth)
            elif max_depth > 0:
                _set_tensors(elmt, all_params, max_depth=max_depth-1)
