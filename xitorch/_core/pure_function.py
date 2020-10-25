import torch
import inspect
from typing import Callable, List, Tuple, Union, Sequence
from xitorch._utils.attr import set_attr, del_attr
from xitorch._utils.unique import Uniquifier
from xitorch._core.editable_module import EditableModule
from contextlib import contextmanager
from abc import abstractmethod

__all__ = ["get_pure_function", "make_sibling"]

############################ functional ###############################
class PureFunction(object):
    """
    PureFunction class wraps methods to make it stateless and expose the pure
    function to take inputs of the original inputs (`params`) and the object's
    states (`objparams`).
    For functions, this class only acts as a thin wrapper.
    """

    def __init__(self, fcntocall: Callable):
        self._state_change_allowed = True
        self._allobjparams = self._get_all_obj_params_init()
        self._uniq = Uniquifier(self._allobjparams)
        self._cur_objparams = self._uniq.get_unique_objs()
        self._fcntocall = fcntocall

        # restore stack stores list of (objparams, identical)
        # everytime the objparams are set, it will store the old objparams
        # and indication if the old and new objparams are identical
        self._restore_stack: List[Tuple[List, bool]] = []

    def __call__(self, *params):
        return self._fcntocall(*params)

    @abstractmethod
    def _get_all_obj_params_init(self):
        pass

    @abstractmethod
    def _set_all_obj_params(self, allobjparams):
        pass

    def objparams(self) -> List:
        return self._cur_objparams

    def set_objparams(self, objparams: List):
        # TODO: check if identical with current object parameters
        identical = _check_identical_objs(objparams, self._cur_objparams)
        self._restore_stack.append((self._cur_objparams, identical))
        if not identical:
            allobjparams = self._uniq.map_unique_objs(objparams)
            self._set_all_obj_params(allobjparams)
            self._cur_objparams = list(objparams)

    def restore_objparams(self):
        old_objparams, identical = self._restore_stack.pop(-1)
        if not identical:
            allobjparams = self._uniq.map_unique_objs(old_objparams)
            self._set_all_obj_params(allobjparams)
            self._cur_objparams = old_objparams

    @contextmanager
    def useobjparams(self, objparams: List):
        if not self._state_change_allowed:
            raise RuntimeError("The state change is disabled")
        try:
            self.set_objparams(objparams)
            yield
        finally:
            self.restore_objparams()

    @contextmanager
    def disable_state_change(self):
        try:
            prev_status = self._state_change_allowed
            self._state_change_allowed = False
            yield
        finally:
            self._state_change_allowed = prev_status

class FunctionPureFunction(PureFunction):
    def _get_all_obj_params_init(self):
        return []

    def _set_all_obj_params(self, objparams):
        pass

class EditableModulePureFunction(PureFunction):
    def __init__(self, obj: EditableModule, method: Callable):
        self.obj = obj
        self.method = method
        super().__init__(method)

    def _get_all_obj_params_init(self) -> List:
        return list(self.obj.getparams(self.method.__name__))

    def _set_all_obj_params(self, allobjparams: List):
        self.obj.setparams(self.method.__name__, *allobjparams)

class TorchNNPureFunction(PureFunction):
    def __init__(self, obj: torch.nn.Module, method: Callable):
        self.obj = obj
        self.method = method
        super().__init__(method)

    def _get_all_obj_params_init(self) -> List:
        # get the tensors in the torch.nn.Module to be used as params
        named_params = list(self.obj.named_parameters())
        if len(named_params) == 0:
            paramnames: List[str] = []
            obj_params: List[Union[torch.Tensor, torch.nn.Parameter]] = []
        else:
            paramnames_temp, obj_params_temp = zip(*named_params)
            paramnames = list(paramnames_temp)
            obj_params = list(obj_params_temp)
        self.names = paramnames
        return obj_params

    def _set_all_obj_params(self, objparams: List):
        for (name, param) in zip(self.names, objparams):
            del_attr(self.obj, name)  # delete required in case the param is not a torch.nn.Parameter
            set_attr(self.obj, name, param)

class SingleSiblingPureFunction(PureFunction):
    def __init__(self, fcn: Callable, fcntocall: Callable):
        self.pfunc = get_pure_function(fcn)
        super().__init__(fcntocall)

    def _get_all_obj_params_init(self) -> List:
        return self.pfunc._get_all_obj_params_init()

    def _set_all_obj_params(self, allobjparams: List):
        self.pfunc._set_all_obj_params(allobjparams)

class MultiSiblingPureFunction(PureFunction):
    def __init__(self, fcns: Sequence[Callable], fcntocall: Callable):
        self.pfuncs = [get_pure_function(fcn) for fcn in fcns]
        self.npfuncs = len(self.pfuncs)
        super().__init__(fcntocall)

    def _get_all_obj_params_init(self) -> List:
        res: List[Union[torch.Tensor, torch.nn.Parameter]] = []
        self.cumsum_idx = [0] * (self.npfuncs + 1)
        for i, pfunc in enumerate(self.pfuncs):
            objparams = pfunc._get_all_obj_params_init()
            res = res + objparams
            self.cumsum_idx[i + 1] = self.cumsum_idx[i] + len(objparams)
        return res

    def _set_all_obj_params(self, allobjparams: List):
        for i, pfunc in enumerate(self.pfuncs):
            pfunc._set_all_obj_params(allobjparams[self.cumsum_idx[i]:self.cumsum_idx[i + 1]])

def _check_identical_objs(objs1: List, objs2: List) -> bool:
    for obj1, obj2 in zip(objs1, objs2):
        if id(obj1) != id(obj2):
            return False
    return True

def get_pure_function(fcn) -> PureFunction:
    """
    Get the pure function form of the function or method ``fcn``.

    Arguments
    ---------
    fcn: function or method
        Function or method to be converted into a ``PureFunction`` by exposing
        the hidden parameters affecting its outputs.

    Returns
    -------
    PureFunction
        The pure function wrapper
    """

    errmsg = "The input function must be a function, a method of " \
        "torch.nn.Module, a method of xitorch.EditableModule, or a sibling method"

    if isinstance(fcn, PureFunction):
        return fcn

    elif inspect.isfunction(fcn) or isinstance(fcn, torch.jit.ScriptFunction):
        return FunctionPureFunction(fcn)

    # if it is a method from an object, unroll the parameters and add
    # the object's parameters as well
    elif inspect.ismethod(fcn) or hasattr(fcn, "__call__"):
        if inspect.ismethod(fcn):
            obj = fcn.__self__
        else:
            obj = fcn
            fcn = fcn.__call__

        if isinstance(obj, EditableModule):
            return EditableModulePureFunction(obj, fcn)
        elif isinstance(obj, torch.nn.Module):
            return TorchNNPureFunction(obj, fcn)
        else:
            raise RuntimeError(errmsg)

    else:
        raise RuntimeError(errmsg)

def make_sibling(*pfuncs) -> Callable[[Callable], PureFunction]:
    """
    Used as a decor to mark the decorated function as a sibling method of the
    input ``pfunc``.
    Sibling method is a method that is virtually belong to the same object, but
    behaves differently.
    Changing the state of the decorated function will also change the state of
    ``pfunc`` and its other siblings.
    """
    if len(pfuncs) == 0:
        raise TypeError("At least 1 function is required as the argument")
    elif len(pfuncs) == 1:
        return lambda fcn: SingleSiblingPureFunction(pfuncs[0], fcntocall=fcn)
    else:
        return lambda fcn: MultiSiblingPureFunction(pfuncs, fcntocall=fcn)


# class PureFunction(object):
#     """
#     PureFunction class wraps methods to make it stateless and expose the pure
#     function to take inputs of the original inputs (`params`) and the object's
#     states (`objparams`).
#     For functions, this class only acts as a thin wrapper.
#     """
#     def __init__(self, fcntocall):
#         self._fcntocall = fcntocall
#         self._state_change_allowed = True
#
#     def __call__(self, *params):
#         return self._fcntocall(*params)
#
#     def _register_allobj_params(self, allobjparams):
#         self._uniq = Uniquifier(allobjparams)
#         self._unique_objparams = self._uniq.get_unique_objs()
#
#     def objparams(self):
#         return self._unique_objparams
#
#     def setobjparams(self, unique_objparams):
#         allobjparams = self._uniq.map_unique_objs(unique_objparams)
#
#     @abstractmethod
#     @contextmanager
#     def _useobjparams(self, objparams):
#         pass
#
#     @contextmanager
#     def useobjparams(self, unique_objparams):
#         if not self._state_change_allowed:
#             raise RuntimeError("The state change is disabled")
#         else:
#             allobjparams = self._uniq.map_unique_objs(unique_objparams)
#             with self._useobjparams(allobjparams):
#                 try: yield
#                 finally: pass
#
#     @contextmanager
#     def disable_state_change(self):
#         try:
#             prev_status = self._state_change_allowed
#             self._state_change_allowed = False
#             yield
#         finally:
#             self._state_change_allowed = prev_status
#
#     # NOTE: please refrain from implementing enable_state_change.
#     # If you want to enable state change, then just remove
#     # `with obj.disable_state_change()` statement in your code
#
# class MultiplePureFunction(PureFunction):
#     def __init__(self, methods, fcntocall=None):
#         self._obj_editors = _get_unique_object_editors(methods)
#         assert len(methods) > 1, "Internal assert error. The class "\
#             "MultiplePureFunction is only for multiple methods. Here we get "\
#             "%d-elements method. Please report this issue."
#         if fcntocall is None:
#             fcntocall = methods[0]
#         super().__init__(fcntocall)
#         self._nobjs = len(self._obj_editors)
#         allobjparams, self._nobj_cumsum = self._getobjparams_init(self._obj_editors)
#         self._nobjparams = len(self._objparams)
#         self._register_allobj_params(allobjparams)
#
#     def _getobjparams_init(self, editors):
#         nobjs = len(editors)
#         nobj_cumsum = [0 for _ in range(nobjs+1)]
#         allobjparams = []
#         for i in range(nobjs):
#             objparams = editors[i].getobjparams()
#             allobjparams = allobjparams + objparams
#             nobj_cumsum[i+1] = nobj_cumsum[i] + len(objparams)
#         return allobjparams, nobj_cumsum
#
#     @contextmanager
#     def _useobjparams(self, allobjparams):
#         mgrs = [None] * self._nobjs
#         for i in range(self._nobjs):
#             editor = self._obj_editors[i]
#             objparams = allobjparams[self._nobj_cumsum[i]:self._nobj_cumsum[i+1]]
#             mgrs[i] = editor.useobjparams(objparams)
#         with contextlib.ExitStack() as stack:
#             try:
#                 for cm in mgrs:
#                     stack.enter_context(cm)
#                 yield
#             finally: pass
#
# class SinglePureFunction(PureFunction):
#     def __init__(self, method, fcntocall=None):
#         if fcntocall is None:
#             fcntocall = method
#         super().__init__(fcntocall)
#         self._obj_editor = _get_object_editor(method)
#         objparams = self._obj_editor.getobjparams()
#         self._register_allobj_params(objparams)
#
#     @contextmanager
#     def _useobjparams(self, objparams):
#         with self._obj_editor.useobjparams(objparams):
#             try: yield
#             finally: pass
#
# class _ObjectEditor(object):
#     def __init__(self, obj, method):
#         self._obj = obj
#         self._method = method
#
#     @property
#     def obj(self):
#         return self._obj
#
#     @property
#     def method(self):
#         return self._method
#
#     @abstractmethod
#     def getobjparams(self) -> List[torch.Tensor]:
#         pass
#
#     @abstractmethod
#     def setobjparams(self, objparams:List[torch.Tensor]):
#         pass
#
#     def restoreobjparams(self, objparams:List[torch.Tensor]):
#         self.setobjparams(objparams)
#
# class _EditableModuleEditor(_ObjectEditor):
#     def getobjparams(self):
#         objparams = self.obj.getparams(self.method.__name__)
#         return objparams
#
#     def setobjparams(self, objparams):
#         self.obj.setparams(self.method.__name__, *objparams)
#
# class _TorchNNEditor(_ObjectEditor):
#     def __init__(self, obj, method):
#         super().__init__(obj, method)
#
#         # get the tensors in the torch.nn.Module to be used as params
#         named_params = list(obj.named_parameters())
#         if len(named_params) == 0:
#             paramnames = []
#             obj_params = []
#         else:
#             paramnames, obj_params = zip(*named_params)
#         self.names = paramnames
#
#     def getobjparams(self):
#         return [get_attr(self.obj, name) for name in self.names]
#
#     def setobjparams(self, objparams):
#         for (name, param) in zip(self.names, objparams):
#             del_attr(self.obj, name) # delete required in case the param is not a torch.nn.Parameter
#             set_attr(self.obj, name, param)
#
#     def restoreobjparams(self, objparams):
#         for (name, param) in zip(self.names, objparams):
#             # delete is not required because the objparams are all torch.nn.Parameter
#             set_attr(self.obj, name, param)
#
# class _FunctionEditor(_ObjectEditor):
#     def getobjparams(self):
#         return []
#
#     def setobjparams(self, objparams):
#         pass
#
# class _PureFunctionEditor(_ObjectEditor):
#     def getobjparams(self):
#         return self.obj.objparams()
#
#     def setobjparams(self, objparams):
#         return self.obj.setobjparams(objparams)
#
# def _get_object_editor(method, return_obj=False):
#     if isinstance(method, PureFunction):
#         editor = _PureFunctionEditor(method, method)
#         obj = method
#
#     elif inspect.isfunction(method) or isinstance(method, torch.jit.ScriptFunction):
#         editor = _FunctionEditor(None, method)
#         obj = None
#
#     # if it is a method from an object, unroll the parameters and add
#     # the object's parameters as well
#     elif inspect.ismethod(method) or hasattr(method, "__call__"):
#         if inspect.ismethod(method):
#             obj = method.__self__
#         else:
#             obj = method
#             method = method.__call__
#
#         if isinstance(obj, EditableModule):
#             editor = _EditableModuleEditor(obj, method)
#         elif isinstance(obj, torch.nn.Module):
#             editor = _TorchNNEditor(obj, method)
#         else:
#             raise RuntimeError("The method must be a method of torch.nn.Module or xitorch.EditableModule")
#
#     else:
#         raise RuntimeError("The fcn must be a function or a method")
#
#     if return_obj:
#         return editor, obj
#     else:
#         return editor
#
# def _get_unique_object_editors(methods):
#     editors = []
#     set_objs = set()
#     for method in methods:
#         editor, obj = _get_object_editor(method, return_obj=True)
#         if obj in set_objs:
#             continue
#         set_objs.add(obj)
#         editors.append(editor)
#     return editors
#
# def make_sibling(*pfuncs) -> Callable[[Callable[...,Any]], PureFunction]:
#     """
#     Used as a decor to mark the decorated function as a sibling method of the
#     input ``pfunc``.
#     Sibling method is a method that is virtually belong to the same object, but
#     behaves differently.
#     Changing the state of the decorated function will also change the state of
#     ``pfunc`` and its other siblings.
#     """
#     if len(pfuncs) == 0:
#         raise TypeError("At least 1 function is required as the argument")
#     elif len(pfuncs) == 1:
#         decor = lambda fcn: SinglePureFunction(pfuncs[0], fcntocall=fcn)
#     else:
#         decor = lambda fcn: MultiplePureFunction(pfuncs, fcntocall=fcn)
#     return decor
#
# def get_pure_function(fcn:Callable[...,Any]) -> PureFunction:
#     return SinglePureFunction(fcn)
