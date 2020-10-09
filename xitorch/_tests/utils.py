import itertools
import torch
import xitorch as xt
import pytest
import argparse
from typing import Mapping, List, ClassVar, Callable, Optional, Tuple

__all__ = ["device_dtype_float_test"]

def device_dtype_float_test(only64:int=False, onlycpu:bool=False,
        additional_kwargs:Mapping[str,List]={},
        skip_fcn:Optional[Callable[...,Tuple[bool,str]]]=None) -> Callable:

    dtypes = [torch.float, torch.float64]
    devices = [torch.device("cpu"), torch.device("cuda")]
    if only64:
        dtypes = [torch.float64]
    if onlycpu or not torch.cuda.is_available():
        devices = [torch.device("cpu")]
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
