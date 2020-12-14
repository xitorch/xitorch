import torch
import pytest
from typing import List
from xitorch._core.editable_module import EditableModule
from xitorch._core.pure_function import get_pure_function
from xitorch._utils.exceptions import GetSetParamsError

##############
# test the assertion with methods with various problems
class ModuleTest(EditableModule):
    def __init__(self, a: torch.Tensor) -> None:
        self.a = a
        self.c = a * 2.
        self.d = a * 3.
        self.e = a + 3.
        self.fint = 1
        self.g = a + 1.
        self.aa = a
        self.aaa = a
        self.dctparams = {
            0: a + 0.1,
            1: a + 1.,
            2: a + 2.,
        }
        self.listparams = [a + 0.12, a + 2., a + 4]

    def method_no_preserve1(self, b: torch.Tensor) -> torch.Tensor:
        # this method changes a parameter
        self.a += b
        return self.a

    def method_no_preserve2(self, b: torch.Tensor) -> torch.Tensor:
        # this method adds a parameter to the object
        self.b = b
        return self.b * 2.0

    def method_dict_correct(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.dctparams[0] + self.dctparams[2]

    def method_dict_missing(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.dctparams[0] + self.dctparams[2]

    def method_dict_excess(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.dctparams[0] + self.dctparams[2]

    def method_list_correct(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.listparams[0] + self.listparams[2]

    def method_list_missing(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.listparams[0] + self.listparams[2]

    def method_list_excess(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.listparams[0] + self.listparams[2]

    def method_duplicate_missing(self, b: torch.Tensor) -> torch.Tensor:
        # `aa` is `a`, but `aa` will be missing in getparamnames
        return self._dummy_fcn(b) + self.aa

    def method_duplicate_excess(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.aa

    def method_duplicate_correct(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self.aa

    def method_correct_getsetparams(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_correct_getsetparams2(self, b: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b) + self._dummy_fcn(b2)

    def method_nontensor_getparams(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_missing_getparams(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def method_excess_getparams(self, b: torch.Tensor) -> torch.Tensor:
        return self._dummy_fcn(b)

    def _dummy_fcn(self, b: torch.Tensor) -> torch.Tensor:
        return self.a + b + self.c + self.d + self.e + self.fint

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "method_no_preserve1":
            return [prefix + "a"]
        elif methodname == "method_no_preserve1":
            return []

        elif methodname == "method_dict_correct":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "dctparams[0]", prefix + "dctparams[2]"]
        elif methodname == "method_dict_missing":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "dctparams[0]"]
        elif methodname == "method_dict_excess":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "dctparams[0]", prefix + "dctparams[1]",
                    prefix + "dctparams[2]"]

        elif methodname == "method_list_correct":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "listparams[0]", prefix + "listparams[2]"]
        elif methodname == "method_list_missing":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "listparams[0]"]
        elif methodname == "method_list_excess":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "listparams[0]", prefix + "listparams[1]",
                    prefix + "listparams[2]"]

        elif methodname == "method_duplicate_correct":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "aa"]
        elif methodname == "method_duplicate_missing":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e"]
        elif methodname == "method_duplicate_excess":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "aa", prefix + "aaa"]

        elif methodname == "method_correct_getsetparams":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e"]
        elif methodname == "method_correct_getsetparams2":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e"]

        elif methodname == "method_nontensor_getparams":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "fint"]
        elif methodname == "method_missing_getparams":
            return [prefix + "a", prefix + "c", prefix + "d"]
        elif methodname == "method_excess_getparams":
            return [prefix + "a", prefix + "c", prefix + "d", prefix + "e",
                    prefix + "g"]
        else:
            raise KeyError("getparams for %s is not implemented" % methodname)


a = torch.tensor([1.])
b = torch.tensor([2.1])
model = ModuleTest(a)

@pytest.mark.filterwarnings("error")
def test_correct():
    correct_methods = {
        "method_correct_getsetparams": (b,),
        "method_correct_getsetparams2": (b, b),
        "method_duplicate_correct": (b,),
        "method_duplicate_missing": (b,),
        "method_duplicate_excess": (b,),
        "method_dict_correct": (b,),
        "method_list_correct": (b,),
    }
    for m in correct_methods:
        model.assertparams(getattr(model, m), *correct_methods[m])

def test_error_getsetparams():
    error_methods = [
        "method_no_preserve1",
        "method_no_preserve2",
        "method_nontensor_getparams",
    ]
    for methodname in error_methods:
        try:
            print(methodname)
            model.assertparams(getattr(model, methodname), b)
            assert False, "A GetSetParamsError must be raised in this case"
        except GetSetParamsError:
            pass

def test_warning_getsetparams():
    warning_methods = [
        "method_missing_getparams",
        "method_excess_getparams",
        "method_dict_missing",
        "method_dict_excess",
        "method_list_missing",
        "method_list_excess",
    ]
    for methodname in warning_methods:
        with pytest.warns(UserWarning):
            model.assertparams(getattr(model, methodname), b)

def test_get_unique_params_leaves():
    # test getuniqueparams in EditableModule

    # redefine the model in this case only
    a = torch.tensor([1.], requires_grad=True)
    model = ModuleTest(a)

    params = model.getuniqueparams(methodname="method_list_correct", onlyleaves=True)
    assert len(params) == 1
    assert params[0] is a

    params = model.getuniqueparams(methodname="method_list_correct")
    assert len(params) == 6

##############
# test the wrap function to make it a functional
def test_edit_simple():
    pfcn = get_pure_function(model.method_correct_getsetparams)
    objparams = pfcn.objparams()
    assert len(objparams) == 4
    newb = torch.tensor([1.])
    newobjparams = [torch.tensor(1.0 * i + 2) for i in range(len(objparams))]
    with pfcn.useobjparams(newobjparams):
        f = pfcn(newb)
    assert torch.allclose(f, f * 0 + 16)  # (1+2+3+4+5+1)

    pfcn2 = get_pure_function(model.method_correct_getsetparams2)
    objparams2 = pfcn2.objparams()
    assert len(objparams2) == 4
    newparams2 = [torch.tensor(1.0 * i + 1) for i in range(2)]
    newobjparams2 = [torch.tensor(1.0 * i + 3) for i in range(len(objparams2))]
    with pfcn2.useobjparams(newobjparams2):
        f2 = pfcn2(*newparams2)
    assert torch.allclose(f2, f2 * 0 + 41)  # (1+3+4+5+6+1) + (2+3+4+5+6+1)

def test_edit_duplicate():
    pfcn = get_pure_function(model.method_duplicate_correct)
    objparams = pfcn.objparams()
    assert len(objparams) == 4
    assert objparams[0] is model.a
    assert objparams[0] is model.aa  # aa is a duplicate of a
    newparams = [torch.tensor(1.0 * i + 1) for i in range(1)]
    newobjparams = [torch.tensor(1.0 * i + 2) for i in range(len(objparams))]
    with pfcn.useobjparams(newobjparams):
        f = pfcn(*newparams)
    assert torch.allclose(f, f * 0 + 18)  # (1+2+3+4+5+1) + 2

def test_edit_dict():
    pfcn = get_pure_function(model.method_dict_correct)
    objparams = pfcn.objparams()
    assert len(objparams) == 6
    assert objparams[-1] is model.dctparams[2]
    assert objparams[-2] is model.dctparams[0]
    newparams = [torch.tensor(1.0 * i + 1) for i in range(1)]
    newobjparams = [torch.tensor(1.0 * i + 2) for i in range(len(objparams))]
    with pfcn.useobjparams(newobjparams):
        f = pfcn(*newparams)
    assert torch.allclose(f, f * 0 + 29)  # (1+2+3+4+5+1) + 6+7

def test_edit_list():
    pfcn = get_pure_function(model.method_list_correct)
    objparams = pfcn.objparams()
    assert len(objparams) == 6
    assert objparams[-1] is model.listparams[2]
    assert objparams[-2] is model.listparams[0]
    newparams = [torch.tensor(1.0 * i + 1) for i in range(1)]
    newobjparams = [torch.tensor(1.0 * i + 2) for i in range(len(objparams))]
    with pfcn.useobjparams(newobjparams):
        f = pfcn(*newparams)
    assert torch.allclose(f, f * 0 + 29)  # (1+2+3+4+5+1) + 6+7
