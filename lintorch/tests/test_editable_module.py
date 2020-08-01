import torch
import pytest
from typing import List
from lintorch.core.editable_module import EditableModule
from lintorch.utils.exceptions import GetSetParamsError

class ModuleTest(EditableModule):
    # this class contains two methods that do not preserve and rest of the methods
    # preserve.
    # for the preserving methods: 1 with unmatched getsetparams,
    #
    def __init__(self, a:torch.Tensor) -> None:
        self.a = a
        self.c = a * 2.
        self.d = a*3.
        self.e = a+3.
        self.fint = 1
        self.g = a + 1.

    def method_no_preserve1(self, b:torch.Tensor) -> torch.Tensor:
        # this method changes a parameter
        self.a += b
        return self.a

    def method_no_preserve2(self, b:torch.Tensor) -> torch.Tensor:
        # this method adds a parameter to the object
        self.b = b
        return self.b * 2.0

    def method_correct_getsetparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_unmatched_getsetparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_nontensor_getparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_missing_getparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_excess_getparams(self, b:torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def _dummy_fcn(self, b:torch.Tensor) -> torch.Tensor:
        return self.a * b + self.c + b * self.d * self.e + self.fint

    def getparams(self, methodname:str) -> List[torch.Tensor]:
        if methodname == "method_no_preserve1":
            return [self.a]
        elif methodname == "method_no_preserve1":
            return []
        elif methodname == "method_correct_getsetparams":
            return [self.a, self.c, self.d, self.e]
        elif methodname == "method_unmatched_getsetparams":
            return [self.a, self.c, self.d, self.e]
        elif methodname == "method_nontensor_getparams":
            return [self.a, self.c, self.d, self.e, self.fint]
        elif methodname == "method_missing_getparams":
            return [self.a, self.c, self.d]
        elif methodname == "method_excess_getparams":
            return [self.a, self.c, self.d, self.e, self.g]
        else:
            raise KeyError("getparams for %s is not implemented" % methodname)

    def setparams(self, methodname:str, *params) -> int:
        if methodname == "method_no_preserve1":
            self.a, = params[:1]
            return 1
        elif methodname == "method_no_preserve2":
            return 0
        elif methodname == "method_correct_getsetparams":
            self.a, self.c, self.d, self.e = params[:4]
            return 4
        elif methodname == "method_unmatched_getsetparams":
            self.a, self.c = params[:2]
            return 2
        elif methodname == "method_nontensor_getparams":
            self.a, self.c, self.d, self.e, self.fint = params[:5]
            return 5
        elif methodname == "method_missing_getparams":
            self.a, self.c, self.d = params[:3]
            return 3
        elif methodname == "method_excess_getparams":
            self.a, self.c, self.d, self.e, self.g = params[:5]
            return 5
        else:
            raise KeyError("setparams for %s is not implemented" % methodname)

a = torch.tensor([1.])
b = torch.tensor([2.])
model = ModuleTest(a)

def test_correct():
    model.assertparams("method_correct_getsetparams", b)

def test_error_getsetparams():
    error_methods = [
        "method_no_preserve1",
        "method_no_preserve2",
        "method_unmatched_getsetparams",
        "method_nontensor_getparams",
        "method_excess_getparams"
    ]
    for methodname in error_methods:
        try:
            print(methodname)
            model.assertparams(methodname, b)
            assert False, "A GetSetParamsError must be raised in this case"
        except GetSetParamsError:
            pass

def test_warning_getsetparams():
    warning_methods = ["method_missing_getparams"]
    for methodname in warning_methods:
        with pytest.warns(UserWarning):
            model.assertparams(methodname, b)
