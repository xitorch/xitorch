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

############################ debugging functions ############################

def list_operating_params(method, *args, **kwargs):
    """
    List the tensors used in executing the method
    """
    obj = method.__self__

    # get all the tensors recursively
    all_tensors, all_names = _get_tensors(obj, prefix="self")

    # copy the tensors and require them to be differentiable
    copy_tensors = [tensor.clone().detach().requires_grad_() for tensor in all_tensors]
    _set_tensors(obj, copy_tensors)

    # run the method and see which one has the gradients
    output = method(*args, **kwargs).sum()
    grad_tensors = torch.autograd.grad(output, copy_tensors, allow_unused=True)

    res = []
    for i, grad in enumerate(grad_tensors):
        if grad is None:
            continue
        res.append(all_names[i])

    # print the results
    res_str = ", ".join(res)
    print("%s: [%s]" % (method.__name__, res_str))

def _get_tensors(obj, prefix):
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
        elif isinstance(elmt, torch.nn.Module):
            new_res, new_names = _get_tensors(elmt, prefix=name)
            res = res + new_res
            names = names + new_names
    return res, names

def _set_tensors(obj, params):
    all_params = copy.copy(params)
    float_type = [torch.float32, torch.float, torch.float64, torch.float16]
    for key in obj.__dict__:
        elmt = obj.__dict__[key]
        if isinstance(elmt, torch.Tensor) and elmt.dtype in float_type:
            obj.__dict__[key] = all_params.pop(0)
        elif isinstance(elmt, torch.nn.Module):
            _set_tensors(elmt, all_params)
