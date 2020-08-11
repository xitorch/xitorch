import torch
import pytest
from typing import List
from lintorch.core.editable_module import EditableModule
from lintorch.utils.exceptions import GetSetParamsError

class ModuleTest(EditableModule):
    def __init__(self, a:torch.Tensor) -> None:
        self.a = a
        self.c = a * 2.
        self.d = a*3.
        self.e = a+3.
        self.fint = 1
        self.g = a + 1.
        self.aa = a
        self.aaa = a

    def method_no_preserve1(self, b:torch.Tensor) -> torch.Tensor:
        # this method changes a parameter
        self.a += b
        return self.a

    def method_no_preserve2(self, b:torch.Tensor) -> torch.Tensor:
        # this method adds a parameter to the object
        self.b = b
        return self.b * 2.0

    def method_duplicate_missing(self, b:torch.Tensor) -> torch.Tensor:
        # `aa` is `a`, but `aa` will be missing in getparamnames
        return self._dummy_fcn(b) + self.aa

    def method_duplicate_excess(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.aa

    def method_duplicate_correct(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.aa

    def method_correct_getsetparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_nontensor_getparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_missing_getparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_excess_getparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def _dummy_fcn(self, b:torch.Tensor) -> torch.Tensor:
        return self.a * b + self.c + b * self.d * self.e + self.fint

    def getparamnames(self, methodname:str, prefix:str="") -> List[str]:
        if methodname == "method_no_preserve1":
            return [prefix+"a"]
        elif methodname == "method_no_preserve1":
            return []
        elif methodname == "method_duplicate_correct":
            return [prefix+"a", prefix+"c", prefix+"d", prefix+"e", prefix+"aa"]
        elif methodname == "method_duplicate_missing":
            return [prefix+"a", prefix+"c", prefix+"d", prefix+"e"]
        elif methodname == "method_duplicate_excess":
            return [prefix+"a", prefix+"c", prefix+"d", prefix+"e", prefix+"aa", prefix+"aaa"]
        elif methodname == "method_correct_getsetparams":
            return [prefix+"a", prefix+"c", prefix+"d", prefix+"e"]
        elif methodname == "method_nontensor_getparams":
            return [prefix+"a", prefix+"c", prefix+"d", prefix+"e", prefix+"fint"]
        elif methodname == "method_missing_getparams":
            return [prefix+"a", prefix+"c", prefix+"d"]
        elif methodname == "method_excess_getparams":
            return [prefix+"a", prefix+"c", prefix+"d", prefix+"e", prefix+"g"]
        else:
            raise KeyError("getparams for %s is not implemented" % methodname)

a = torch.tensor([1.])
b = torch.tensor([2.])
model = ModuleTest(a)

def test_correct():
    correct_methods = [
        "method_correct_getsetparams",
        "method_duplicate_correct",
    ]
    for m in correct_methods:
        model.assertparams(m, b)

def test_error_getsetparams():
    error_methods = [
        "method_no_preserve1",
        "method_no_preserve2",
        "method_nontensor_getparams",
    ]
    for methodname in error_methods:
        try:
            print(methodname)
            model.assertparams(methodname, b)
            assert False, "A GetSetParamsError must be raised in this case"
        except GetSetParamsError:
            pass

def test_warning_getsetparams():
    warning_methods = [
        "method_missing_getparams",
        "method_excess_getparams",
        "method_duplicate_missing",
        "method_duplicate_excess",
    ]
    for methodname in warning_methods:
        with pytest.warns(UserWarning):
            model.assertparams(methodname, b)
