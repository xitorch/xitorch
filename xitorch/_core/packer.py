from __future__ import annotations
from typing import Any, Optional, List, Tuple, Dict
from copy import deepcopy, copy
import torch

__all__ = ["Packer"]

class Packer(object):
    """
    Packer is an object that could extract the tensors in a structure and
    rebuild the structure from the given tensors.
    This object preserves the structure of the object by performing the deepcopy
    of the object, except for the tensor.

    Arguments
    ---------
    obj: Any
        Any structure object that contains tensors.

    Example
    -------
    .. testsetup::

        import torch
        import xitorch

    .. doctest::

        >>> a = torch.tensor(1.0)
        >>> obj = {
        ...     "a": a,
        ...     "b": a * 3,
        ...     "c": a,
        ... }
        >>> packer = xitorch.Packer(obj)
        >>> tensors = packer.get_param_tensor_list()
        >>> print(tensors)
        [tensor(1.), tensor(3.)]
        >>> new_tensors = [torch.tensor(2.0), torch.tensor(4.0)]
        >>> new_obj = packer.construct_from_tensor_list(new_tensors)
        >>> print(new_obj)
        {'a': tensor(2.), 'b': tensor(4.), 'c': tensor(2.)}
    """
    def __init__(self, obj: Any):
        # deep copy the object, except the tensors
        tensor_lists = _extract_tensors(obj)
        memo = {id(t): t for t in tensor_lists}
        self._tensor_memo = copy(memo)  # shallow copy
        self._obj = deepcopy(obj, memo)

        # caches
        self._params_tensor_list: Optional[List[torch.Tensor]] = tensor_lists
        self._unique_params_idxs: Optional[List[int]] = None
        self._unique_inverse_idxs: Optional[List[int]] = None
        self._unique_tensor_shapes: Optional[List[torch.Size]] = None
        self._tensor_shapes: Optional[List[torch.Size]] = None
        self._unique_tensor_numels: Optional[List[int]] = None
        self._unique_tensor_numel_tot: Optional[int] = None
        self._tensor_numels: Optional[List[int]] = None
        self._tensor_numel_tot: Optional[int] = None

    def get_param_tensor_list(self, unique: bool = True) -> List[torch.Tensor]:
        """
        Returns the list of tensors contained in the object. It will traverse
        down the object via elements for list, values for dictionary, or
        ``__dict__`` for object that has ``__dict__`` attribute.

        Arguments
        ---------
        unique: bool
            If True, then only returns the unique tensors. Otherwise, duplicates
            can also be returned.

        Returns
        -------
        list of torch.Tensor
            List of tensors contained in the object.
        """
        # get the params tensor list
        if self._params_tensor_list is not None:
            # get it from cache if available
            params_tensors = self._params_tensor_list
        else:
            params_tensors = _extract_tensors(self._obj)
            self._params_tensor_list = params_tensors

        # only take the unique tensors if required
        if unique:
            if self._unique_params_idxs is not None:
                unique_idxs = self._unique_params_idxs
                unique_inverse = self._unique_inverse_idxs
            else:
                unique_idxs, unique_inverse = _get_unique_idxs(params_tensors)
                self._unique_params_idxs = unique_idxs
                self._unique_inverse_idxs = unique_inverse

            params_tensors = [params_tensors[i] for i in unique_idxs]

        if unique:
            self._unique_tensor_shapes = [p.shape for p in params_tensors]
        else:
            self._tensor_shapes = [p.shape for p in params_tensors]

        return params_tensors

    def get_param_tensor(self, unique: bool = True) -> Optional[torch.Tensor]:
        """
        Returns the tensor parameters as a single tensor. This can be used,
        for example, if there are multiple parameters to be optimized using
        ``xitorch.optimize.minimize``.

        Arguments
        ---------
        unique: bool
            If True, then only returns the tensor from unique tensors list.
            Otherwise, duplicates can also be returned.

        Returns
        -------
        torch.Tensor or None
            The parameters of the object in a single tensor or None if there
            is no tensor contained in the object.
        """
        params = self.get_param_tensor_list(unique=unique)
        if len(params) == 0:
            return None
        else:
            if unique:
                self._unique_tensor_numels = [p.numel() for p in params]
                self._unique_tensor_numel_tot = sum(self._unique_tensor_numels)
            else:
                self._tensor_numels = [p.numel() for p in params]
                self._tensor_numel_tot = sum(self._tensor_numels)
            if len(params) == 1:
                return params[0]
            else:
                tparam = torch.cat([p.reshape(-1) for p in params])
                return tparam

    def construct_from_tensor_list(self, tensors: List[torch.Tensor], unique: bool = True) -> Any:
        """
        Construct the object from the tensor list and returns the object structure
        with the new tensors. Executing this does not change the state of the
        Packer object.

        Arguments
        ---------
        tensors: list of torch.Tensor
            The tensor parameters to be filled into the object.
        unique: bool
            Indicating if the tensor list ``tensors`` is from the unique
            parameters of the object.

        Returns
        -------
        Any
            A new object with the same structure as the input to ``__init__``
            object except the tensor is changed according to ``tensors``.
        """
        if unique:
            tensor_shapes = self._unique_tensor_shapes
        else:
            tensor_shapes = self._tensor_shapes

        if tensor_shapes is None:
            raise RuntimeError("Please execute self.get_param_tensor_list(%s) first" % str(unique))
        else:
            # make sure the length matches
            if len(tensor_shapes) != len(tensors):
                raise RuntimeError("Mismatch length of the tensors")

            if len(tensor_shapes) == 0:
                return self._obj

            # check the tensor shapes
            for i, (tens, shape) in enumerate(zip(tensors, tensor_shapes)):
                if tens.shape != shape:
                    msg = "The tensors[%d] has mismatch shape from the original. Expected: %s, got: %s" % \
                          (i, tens.shape, shape)
                    raise RuntimeError(msg)

            # duplicate the tensors if the input is unique list of tensors
            if unique:
                assert self._unique_inverse_idxs, "Please report to Github"
                tensors = [tensors[self._unique_inverse_idxs[i]] for i in range(len(self._unique_inverse_idxs))]
            else:
                # _put_tensors will change the tensors, so this is just to preserve
                # the input
                tensors = copy(tensors)

            # deepcopy the object, except the tensors
            memo = copy(self._tensor_memo)
            new_obj = deepcopy(self._obj, memo)
            new_obj = _put_tensors(new_obj, tensors)

            return new_obj

    def construct_from_tensor(self, a: torch.Tensor, unique: bool = True) -> Any:
        """
        Construct the object from the single tensor (i.e. it is the parameters
        tensor merged into a single tensor) and returns the object structure
        with the new tensor. Executing this does not change the state of the
        Packer object.

        Arguments
        ---------
        a: torch.Tensor
            The single tensor parameter to be filled.
        unique: bool
            Indicating if the tensor ``a`` is from the unique parameters of the
            object.

        Returns
        -------
        Any
            A new object with the same structure as the input to ``__init__``
            object except the tensor is changed according to ``a``.
        """
        if unique:
            tensor_shapes = self._unique_tensor_shapes
            tensor_numel_tot = self._unique_tensor_numel_tot
            tensor_numels = self._unique_tensor_numels
        else:
            tensor_shapes = self._tensor_shapes
            tensor_numel_tot = self._tensor_numel_tot
            tensor_numels = self._tensor_numels

        if tensor_shapes is None:
            raise RuntimeError("Please execute self.get_param_tensor(%s) first" % str(unique))
        elif len(tensor_shapes) == 0:
            return self._obj
        else:
            assert tensor_numel_tot is not None, "Please report to Github"
            assert tensor_numels is not None, "Please report to Github"
            if a.numel() != tensor_numel_tot:
                msg = "The number of element does not match. Expected: %d, got: %d" % \
                      (tensor_numel_tot, a.numel())
                raise RuntimeError(msg)

            if len(tensor_numels) == 1:
                params: List[torch.Tensor] = [a]
            else:
                # reshape the parameters
                ioffset = 0
                params = []
                for i in range(len(tensor_numels)):
                    p = a[ioffset:ioffset + tensor_numels[i]].reshape(tensor_shapes[i])
                    ioffset += tensor_numels[i]
                    params.append(p)

            return self.construct_from_tensor_list(params, unique=unique)

def _extract_tensors(b: Any) -> List[torch.Tensor]:
    # extract all the tensors from the given object
    # this function traverses down the object to collect all the tensors

    res: List[torch.Tensor] = []
    if isinstance(b, torch.Tensor):
        res.append(b)
    elif isinstance(b, list):
        for elmt in b:
            res.extend(_extract_tensors(elmt))
    elif isinstance(b, dict):
        for elmt in b.values():
            res.extend(_extract_tensors(elmt))
    elif hasattr(b, "__dict__"):
        for elmt in b.__dict__.values():
            res.extend(_extract_tensors(elmt))
    return res

def _put_tensors(b: Any, tensors: List) -> Any:
    # put the tensors recursively in the object, with the same order as
    # _extract_tensors.
    # the tensors will be changed in this class, so make sure to have
    # a shallow copy if you want to preserve your input
    if isinstance(b, torch.Tensor):
        b = tensors.pop(0)
    elif isinstance(b, list):
        for i, elmt in enumerate(b):
            b[i] = _put_tensors(elmt, tensors)
    elif isinstance(b, dict):
        for key, elmt in b.items():
            b[key] = _put_tensors(elmt, tensors)
    elif hasattr(b, "__dict__"):
        for key, elmt in b.__dict__.items():
            b.__dict__[key] = _put_tensors(elmt, tensors)
    return b

def _get_unique_idxs(b: List) -> Tuple[List[int], List[int]]:
    # get unique indices based on the ids of the b's elements
    # and the index for inversing the unique process

    ids_list = [id(bb) for bb in b]
    unique_ids: Dict[int, int] = {}
    unique_idxs: List[int] = []
    unique_inverse: List[int] = []
    for i, idnum in enumerate(ids_list):
        if idnum in unique_ids:
            unique_inverse.append(unique_ids[idnum])
        else:
            unique_ids[idnum] = len(unique_idxs)
            unique_idxs.append(i)
            unique_inverse.append(unique_ids[idnum])
    return unique_idxs, unique_inverse
