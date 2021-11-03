import warnings
import itertools
import pytest
import torch
from xitorch._utils.unique import Uniquifier
from xitorch._utils.decorators import deprecated
from xitorch._utils.tensor import create_random_square_matrix, \
    create_random_ortho_matrix

@deprecated("06 Oct 2020")
def deprfunc():
    return 3

@deprecated("06 Oct 2020")
class DeprClass():
    def __init__(self):
        self.a = 3

class DeprMethod():
    def __init__(self):
        self.a = 4

    @deprecated("06 Oct 2020")
    def method(self):
        return self.a + 2

def test_uniquifier():
    obj1 = [1]
    obj2 = [2]
    obj3 = [3]
    obj4 = [4]
    obj5 = [5]
    objs = [obj1, obj2, obj1, obj2, obj3, obj3]
    objsA = [obj2, obj1, obj2, obj1, obj4, obj4]

    uniq = Uniquifier(objs)
    unique_objs = uniq.get_unique_objs()
    assert len(unique_objs) == 3
    assert unique_objs[0] is obj1
    assert unique_objs[1] is obj2
    assert unique_objs[2] is obj3

    unique_objsA = uniq.get_unique_objs(objsA)
    assert len(unique_objsA) == 3
    assert unique_objsA[0] is obj2
    assert unique_objsA[1] is obj1
    assert unique_objsA[2] is obj4

    unique_objs2 = [obj3, obj4, obj5]
    objs2 = uniq.map_unique_objs(unique_objs2)
    assert len(objs2) == len(objs)
    assert objs2[0] is obj3
    assert objs2[1] is obj4
    assert objs2[2] is obj3
    assert objs2[3] is obj4
    assert objs2[4] is obj5
    assert objs2[5] is obj5

def test_uniquifier_error():
    obj1 = [1]
    obj2 = [2]
    obj3 = [3]
    obj4 = [4]
    objs = [obj1, obj2, obj1, obj1, obj2]
    objs2 = [obj1, obj2, obj1, obj1, obj2, obj2]

    uniq = Uniquifier(objs)
    try:
        unique_objs = uniq.get_unique_objs(objs2)
        assert False, "Expected a RuntimeError"
    except RuntimeError:
        pass

    try:
        objs3 = uniq.map_unique_objs(objs)
        assert False, "Expected a RuntimeError"
    except RuntimeError:
        pass

def test_deprecated_func():
    with warnings.catch_warnings(record=True) as w:
        a = deprfunc()
        assert len(w) == 1
        msg = str(w[0].message)
        assert "deprfunc" in msg
        assert "deprecated" in msg
        assert "06 Oct 2020" in msg

def test_deprecated_class():
    with warnings.catch_warnings(record=True) as w:
        a = DeprClass()
        assert len(w) == 1
        msg = str(w[0].message)
        assert "DeprClass" in msg
        assert "deprecated" in msg
        assert "06 Oct 2020" in msg

def test_deprecated_method():
    with warnings.catch_warnings(record=True) as w:
        a = DeprMethod()
        b = a.method()
        assert len(w) == 1
        msg = str(w[0].message)
        assert "DeprMethod.method" in msg
        assert "deprecated" in msg
        assert "06 Oct 2020" in msg

@pytest.mark.parametrize(
    "is_hermitian,minmaxeival",
    itertools.product([False, True], [(-1.0, 1.0), (0.0, 1.0), (0.5, 1.0)])
)
def test_create_random_matrix(is_hermitian, minmaxeival):
    n = 100
    min_eival, max_eival = minmaxeival
    a = create_random_square_matrix(
        n, is_hermitian=is_hermitian,
        min_eival=min_eival, max_eival=max_eival)
    assert a.shape[0] == n
    assert a.shape[1] == n
    assert a.dtype == torch.float64
    dtype = a.dtype
    if is_hermitian:
        assert torch.allclose(a, a.transpose(-2, -1))
        eivals = torch.linalg.eigvalsh(a)
    else:
        eivals = torch.linalg.eigvals(a)
        assert torch.allclose(torch.imag(eivals), torch.imag(eivals) * 0, atol=1e-4)
        eivals = torch.real(eivals)
    assert torch.allclose(eivals.min(), torch.tensor(min_eival, dtype=dtype),
                          atol=1e-4)
    assert torch.allclose(eivals.max(), torch.tensor(max_eival, dtype=dtype),
                          atol=1e-4)

def test_create_random_ortho_matrix():
    n = 100
    a = create_random_ortho_matrix(n)
    eye = torch.eye(n, dtype=a.dtype)
    assert torch.allclose(torch.matmul(a.transpose(-2, -1), a), eye)
