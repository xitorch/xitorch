import torch
from dataclasses import dataclass
from xitorch._core.packer import Packer

@dataclass
class MyObj:
    a: torch.Tensor
    b: torch.Tensor
    c: int

def test_packer_tensor():
    # test if the object to be packed is a tensor
    a = torch.tensor(1.5)
    b = torch.tensor(2.0)
    b2 = torch.tensor([1.0, 1.0])
    b3 = torch.tensor([4.0])
    packer = Packer(a)

    for unique in [True, False]:
        print(unique)

        # test get tensors
        tensors = packer.get_param_tensor_list(unique=unique)
        assert len(tensors) == 1
        assert tensors[0] is a

        # test set tensors and check if the packer's state is unchanged
        obj = packer.construct_from_tensor_list([b], unique=unique)
        assert obj is b
        assert packer.get_param_tensor_list(unique=unique)[0] is a

        # get the single param tensor
        ptensor = packer.get_param_tensor(unique=unique)
        assert ptensor is a

        # set tensor from a single param
        obj = packer.construct_from_tensor(b, unique=unique)
        assert obj is b
        assert packer.get_param_tensor(unique=unique) is a

        # test if the input has mismatched numel
        try:
            obj = packer.construct_from_tensor(b2, unique=unique)
            assert False
        except RuntimeError as e:
            pass

        # test if the input has mismatched shapes
        try:
            obj = packer.construct_from_tensor(b3, unique=unique)
            assert False
        except RuntimeError as e:
            pass

def test_packer_list_tensors():
    # test if the object is a list of tensors
    a = torch.tensor(1.0)
    b = torch.tensor(2.0)
    obj0 = [a, a, b, a]
    new_tlist = [b, b, a, b]
    packer = Packer(obj0)

    # get tensors
    tensors = packer.get_param_tensor_list(unique=False)
    assert len(tensors) == len(obj0)
    for i in range(len(tensors)):
        assert tensors[i] is obj0[i]

    # get unique tensors (also test if the default is unique=True)
    tensors = packer.get_param_tensor_list()
    assert len(tensors) == 2
    assert tensors[0] == a
    assert tensors[1] == b

    # construct the object from new list
    obj1 = packer.construct_from_tensor_list(new_tlist, unique=False)
    assert len(obj1) == len(obj0)
    assert obj1[0] is b
    assert obj1[1] is b
    assert obj1[2] is a
    assert obj1[3] is b

    # construct from the unique list
    obj2 = packer.construct_from_tensor_list([b, a])
    assert len(obj2) == len(obj0)
    assert obj2[0] is b
    assert obj2[1] is b
    assert obj2[2] is a
    assert obj2[3] is b

    # test immutability by repeating one of the get params test above
    tensors = packer.get_param_tensor_list()
    assert len(tensors) == 2
    assert tensors[0] == a
    assert tensors[1] == b

    # get one tensor as a param
    tparam = packer.get_param_tensor(unique=False)
    assert tparam.numel() == 4
    assert torch.allclose(tparam, torch.tensor([1., 1., 2., 1.]))
    tparam = packer.get_param_tensor()
    assert tparam.numel() == 2
    assert torch.allclose(tparam, torch.tensor([1., 2.]))

    # set the one tensor param
    obj1 = packer.construct_from_tensor(torch.tensor([2., 3., 1., 4.]), unique=False)
    assert len(obj1) == 4
    assert float(obj1[0].item()) == 2.0
    assert float(obj1[1].item()) == 3.0
    assert float(obj1[2].item()) == 1.0
    assert float(obj1[3].item()) == 4.0

    obj1 = packer.construct_from_tensor(torch.tensor([2., 3.]), unique=True)
    assert len(obj1) == 4
    assert float(obj1[0].item()) == 2.0
    assert float(obj1[1].item()) == 2.0
    assert float(obj1[2].item()) == 3.0
    assert float(obj1[3].item()) == 2.0

    # test immutability
    obj0.append(a)  # change the object, and see if packer changes or not
    tparam = packer.get_param_tensor(unique=False)
    assert tparam.numel() == 4
    assert torch.allclose(tparam, torch.tensor([1., 1., 2., 1.]))
    tparam = packer.get_param_tensor()
    assert tparam.numel() == 2
    assert torch.allclose(tparam, torch.tensor([1., 2.]))

    # test shape mismatch
    try:
        obj2 = packer.construct_from_tensor_list([b, a, b, b.unsqueeze(-1)], unique=False)
        assert False
    except RuntimeError:
        pass

def test_packer_null():
    # test if the packer object contains no tensors
    a = [1, 2, 3]
    packer = Packer(a)

    assert packer.get_param_tensor(unique=False) is None
    assert packer.get_param_tensor(unique=True) is None
    assert packer.get_param_tensor_list(unique=False) == []
    assert packer.get_param_tensor_list(unique=True) == []

    obj = packer.construct_from_tensor(None)
    assert obj == a
    obj2 = packer.construct_from_tensor_list([])
    assert obj2 == a

def test_packer_complex_structure():
    a = torch.tensor([1., 2.]).requires_grad_()
    b = torch.tensor([4.])
    c = torch.cat((a, b))
    a2 = a * 2
    b2 = b * 2
    c2 = c * 2
    a3 = a * 3
    myobj = MyObj(a=a, b=b, c=3)
    obj0 = {
        "lst": [myobj, c],
        "a": a,
    }
    packer = Packer(obj0)

    # test if error is raised to construct before calling get tensors
    try:
        tparam_new = torch.tensor([1., 1., 1., 1., 1., 1.])  # numel == 6 for unique
        packer.construct_from_tensor(tparam_new)
        assert False
    except RuntimeError as e:
        assert "Please execute" in str(e)

    try:
        tparams = [a2, b2, c2, a2]
        packer.construct_from_tensor_list(tparams, unique=False)
        assert False
    except RuntimeError as e:
        assert "Please execute" in str(e)

    # get tensors
    tensors = packer.get_param_tensor_list(unique=False)
    assert len(tensors) == 4  # (a, b, c, a)
    assert tensors[0] is a
    assert tensors[1] is b
    assert tensors[2] is c
    assert tensors[3] is a
    tensors = packer.get_param_tensor_list()
    assert len(tensors) == 3  # (a, b, c)

    tparam = packer.get_param_tensor(unique=False)
    assert tparam.numel() == (a.numel() * 2 + b.numel() + c.numel())
    tparam = packer.get_param_tensor()
    assert tparam.numel() == (a.numel() + b.numel() + c.numel())

    # test construct object
    obj1 = packer.construct_from_tensor_list([a2, b2, c2, a3], unique=False)
    assert obj1["a"] is a3
    assert obj1["lst"][1] is c2
    assert obj1["lst"][0].a is a2
    assert obj1["lst"][0].b is b2
    assert obj1["lst"][0].c == 3
    myobj.c = 4  # test immutability
    assert obj0["lst"][0].c == 4  # make sure the obj0 is changed
    # now make sure the object inside packer is not changed
    obj1 = packer.construct_from_tensor_list([a2, b2, c2])
    assert obj1["a"] is a2
    assert obj1["lst"][1] is c2
    assert obj1["lst"][0].a is a2
    assert obj1["lst"][0].b is b2
    assert obj1["lst"][0].c == 3

    # test immutability of the packer object
    tensors = packer.get_param_tensor_list(unique=False)
    assert len(tensors) == 4  # (a, b, c, a)
    assert tensors[0] is a
    assert tensors[1] is b
    assert tensors[2] is c
    assert tensors[3] is a

    # construct object from a single parameter
    tparam = packer.get_param_tensor()
    obj1 = packer.construct_from_tensor(tparam * 4)
    assert torch.allclose(obj1["a"], a * 4)
    assert torch.allclose(obj1["lst"][1], c * 4)
    assert torch.allclose(obj1["lst"][0].a, a * 4)
    assert torch.allclose(obj1["lst"][0].b, b * 4)
    assert obj1["lst"][0].c == 3
    # make sure the parameters in obj1 is in the correct graph with a
    grad_a, = torch.autograd.grad(obj1["a"].sum(), a, retain_graph=True)
    assert torch.allclose(grad_a, grad_a * 0 + 4)

    # test error mismatch numel
    try:
        tparam_new = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        packer.construct_from_tensor(tparam_new)
        assert False
    except RuntimeError as e:
        assert "match" in str(e)
