import gc
import itertools
import torch
import pytest
from typing import Mapping, List, Callable, Optional, Tuple

__all__ = ["device_dtype_float_test", "assert_no_memleak"]

def device_dtype_float_test(only64: int = False, onlycpu: bool = False,
                            additional_kwargs: Mapping[str, List] = {},
                            skip_fcn: Optional[Callable[..., Tuple[bool, str]]] = None,
                            include_complex: bool = False) -> Callable:

    # test function decorator that will setup the dtype, device, and some additional
    # kwargs values to the arguments.
    # the skip_fcn is a function that receives the same argument as the decorated
    # function and should return a tuple of bool and string.
    # the first returned value (i.e. a bool) is whether to skip that parameters
    # and the second one is the reason
    dtypes = [torch.float, torch.float64]
    devices = [torch.device("cpu"), torch.device("cuda")]
    if only64:
        dtypes = [torch.float64]
    if onlycpu or not torch.cuda.is_available():
        devices = [torch.device("cpu")]
    if include_complex:
        dtypes.extend([_get_complex_dtype(dt) for dt in dtypes])

    kwargs_vals = additional_kwargs.values()
    argnames = ",".join(["dtype", "device"] + list(additional_kwargs.keys()))
    iters = itertools.product(dtypes, devices, *kwargs_vals)

    # decide which one to skip
    if skip_fcn is None:
        params = [*iters]
    else:
        params = []
        for p in iters:
            skip, reason = skip_fcn(*p)
            param = pytest.param(*p, marks=pytest.mark.skipif(skip, reason=reason))
            params.append(param)
    return pytest.mark.parametrize(argnames, params)

# memory test functions
def assert_no_memleak(fcn: Callable, strict: bool = True, gccollect: bool = False):
    """
    Assert no memory leak when calling the function.

    Arguments
    ---------
    fcn: Callable
        A function with no input and output to be checked.
    strict: bool
        If True, then there must be no additional tensor allocated after it
        exits the function.
    gccollect: bool
        If True, then manually apply ``gc.collect()`` after the function
        execution.

    Exceptions
    ----------
    AssertionError
        Raised if there is an indication of memory leak in the function.
    """
    size0 = _get_tensor_memory()
    ntries = 10
    if strict:
        fcn()
        if gccollect:
            gc.collect()
        size = _get_tensor_memory()
        if size0 != size:
            _show_memsize(fcn, ntries, gccollect=gccollect)
        assert size0 == size
    else:
        raise NotImplementedError("Option non-strict memory leak checking has not been implemented")

def _get_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Returns the complex data type that corresponds to the input data type.

    Arguments
    ---------
    dtype: torch.dtype
        Real number data type

    Returns
    -------
    torch.dtype
        The complex data type
    """
    if dtype == torch.float32:
        return torch.complex64
    elif dtype == torch.float64:
        return torch.complex128
    else:
        raise RuntimeError("Datatype %s has no complex datatype" % dtype)

def _show_memsize(fcn, ntries: int = 10, gccollect: bool = False):
    # show the memory growth
    for i in range(ntries):
        fcn()
        if gccollect:
            gc.collect()
        size = _get_tensor_memory()
        print("%3d iteration: %.7f MiB of tensors" % (i + 1, size))

def _get_tensor_memory() -> float:
    # obtain the total memory occupied by torch.Tensor in the garbage collector
    # (units in MiB)

    # obtaining all the tensor objects from the garbage collector
    tensor_objs = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]

    # iterate each tensor objects uniquely and calculate the total storage
    visited_data = set([])
    total_mem = 0.0
    for tensor in tensor_objs:
        if tensor.is_sparse:
            continue

        # check if it has been visited
        storage = tensor.storage()
        data_ptr = storage.data_ptr()  # type: ignore
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        # calculate the storage occupied
        numel = storage.size()
        elmt_size = storage.element_size()
        mem = numel * elmt_size / (1024 * 1024)  # in MiB

        total_mem += mem

    return total_mem
