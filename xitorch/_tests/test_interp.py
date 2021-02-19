import warnings
import torch
from torch.autograd import gradcheck, gradgradcheck
from xitorch.interpolate.interp1 import Interp1D
from xitorch._tests.utils import device_dtype_float_test

@device_dtype_float_test(only64=True, additional_kwargs={
    "bc_type": ["clamped", "natural", "not-a-knot", "periodic", None],
    "scramble": [False, True]
})
def test_interp1_cspline(dtype, device, bc_type, scramble):
    dtype_device_kwargs = {"dtype": dtype, "device": device}
    x = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], **dtype_device_kwargs).requires_grad_()
    if bc_type != "periodic":
        y1 = torch.tensor([1.0, 1.5, 2.1, 1.1, 2.3, 2.5], **dtype_device_kwargs).requires_grad_()
        y2 = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 2.5],
                           [0.8, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()
    else:
        y1 = torch.tensor([1.0, 1.5, 2.1, 1.1, 2.3, 1.0], **dtype_device_kwargs).requires_grad_()
        y2 = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 1.0],
                           [0.8, 1.2, 2.2, 0.4, 3.2, 0.8]], **dtype_device_kwargs).requires_grad_()

    # points are well inside to avoid extrapolation in numerical gradient calculations
    xq1 = torch.linspace(0.05, 0.95, 10, **dtype_device_kwargs)
    xq2 = torch.linspace(0.05, 0.95, 4, **dtype_device_kwargs)

    scramble = scramble and bc_type != "periodic"
    if scramble:
        idx1 = torch.randperm(len(xq1))
        idx2 = torch.randperm(len(xq2))
        xq1 = xq1[..., idx1]
        xq2 = xq2[..., idx2]
    xq1 = xq1.requires_grad_()
    xq2 = xq2.requires_grad_()

    # true results (obtained from scipy.interpolate.CubicSpline)
    # from scipy.interpolate import CubicSpline
    # print("yq11:", CubicSpline(x.detach(), y1.detach(), bc_type=bc_type)(xq1.detach()))
    # print("yq12:", CubicSpline(x.detach(), y1.detach(), bc_type=bc_type)(xq2.detach()))
    # print("yq21:", CubicSpline(x.detach(), y2[1].detach(), bc_type=bc_type)(xq1.detach()))
    # print("yq22:", CubicSpline(x.detach(), y2[1].detach(), bc_type=bc_type)(xq2.detach()))

    # get the y_trues from scipy
    if bc_type == "clamped":
        yq11_true = torch.tensor([1.01599131, 1.23547394, 1.85950467, 2.02868906, 1.37102567, 1.04108172,
                                  1.42061722, 2.04849297, 2.4435166, 2.5061722],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([1.01599131, 2.02868906, 1.42061722, 2.5061722], **dtype_device_kwargs)
        yq21_true = torch.tensor([[1.01599131, 1.23547394, 1.85950467, 2.02868906, 1.37102567, 1.04108172,
                                   1.42061722, 2.04849297, 2.4435166, 2.5061722],
                                  [0.76740145, 0.85220436, 1.79469225, 2.01628631, 0.78122407, 0.53357346,
                                   1.80606846, 3.07316928, 2.80705394, 1.48568465]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[1.01599131, 2.02868906, 1.42061722, 2.5061722],
                                  [0.76740145, 2.01628631, 1.80606846, 1.48568465]],
                                 **dtype_device_kwargs)
    elif bc_type == "not-a-knot" or bc_type is None:  # default choice
        yq11_true = torch.tensor([0.66219741, 1.06231845, 1.8959342, 2.01058952, 1.36963168, 1.02084725,
                                  1.33918614, 1.97824847, 2.56027129, 2.70749165],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([0.66219741, 2.01058952, 1.33918614, 2.70749165], **dtype_device_kwargs)
        yq21_true = torch.tensor([[0.66219741, 1.06231845, 1.8959342, 2.01058952, 1.36963168, 1.02084725,
                                   1.33918614, 1.97824847, 2.56027129, 2.70749165],
                                  [-0.01262521, 0.47242487, 1.87087507, 1.99610601, 0.81846828, 0.39785058,
                                   1.33699082, 2.68769477, 3.43433639, 2.56128965]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[0.66219741, 2.01058952, 1.33918614, 2.70749165],
                                  [-0.01262521, 1.99610601, 1.33699082, 2.56128965]],
                                 **dtype_device_kwargs)
    elif bc_type == "natural":
        yq11_true = torch.tensor([1.03045416, 1.24263582, 1.85784168, 2.03025785, 1.37277695, 1.03808008,
                                  1.41177844, 2.04167374, 2.45428693, 2.52449066],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([1.03045416, 2.03025785, 1.41177844, 2.52449066], **dtype_device_kwargs)
        yq21_true = torch.tensor([[1.03045416, 1.24263582, 1.85784168, 2.03025785, 1.37277695, 1.03808008,
                                   1.41177844, 2.04167374, 2.45428693, 2.52449066],
                                  [0.70073217, 0.82102504, 1.79853565, 2.02728778, 0.8104202, 0.46318855,
                                   1.57916384, 2.89143794, 3.09930603, 1.98521859]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[1.03045416, 2.03025785, 1.41177844, 2.52449066],
                                  [0.70073217, 2.02728778, 1.57916384, 1.98521859]],
                                 **dtype_device_kwargs)
    elif bc_type == "periodic":
        yq11_true = torch.tensor([0.88184647, 1.16754002, 1.87806756, 1.99916778, 1.3241823, 1.13211374,
                                  1.69017244, 2.25696675, 2.09041608, 1.31247223],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([0.88184647, 1.99916778, 1.69017244, 1.31247223], **dtype_device_kwargs)
        yq21_true = torch.tensor([[0.88184647, 1.16754002, 1.87806756, 1.99916778, 1.3241823, 1.13211374,
                                   1.69017244, 2.25696675, 2.09041608, 1.31247223],
                                  [0.46559344, 0.70408188, 1.82662341, 1.99677022, 0.77170332, 0.52939286,
                                   1.76540093, 3.03216372, 2.8731096, 1.44347038]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[0.88184647, 1.99916778, 1.69017244, 1.31247223],
                                  [0.46559344, 1.99677022, 1.76540093, 1.44347038]],
                                 **dtype_device_kwargs)

    if scramble:
        yq11_true = yq11_true[..., idx1]
        yq12_true = yq12_true[..., idx2]
        yq21_true = yq21_true[..., idx1]
        yq22_true = yq22_true[..., idx2]

    def interp(x, y, xq):
        return Interp1D(x, y, method="cspline", bc_type=bc_type)(xq)

    yq11 = interp(x, y1, xq1)
    yq12 = interp(x, y1, xq2)
    yq21 = interp(x, y2, xq1)
    yq22 = interp(x, y2, xq2)
    # import matplotlib.pyplot as plt
    # from scipy.interpolate import CubicSpline
    # xx = torch.linspace(0, 1, 1000, **dtype_device_kwargs)
    # xx2 = torch.linspace(-1, 2, 1000, **dtype_device_kwargs)
    # plt.plot(xx2, interp(x, y1, xx2).detach().numpy())
    # plt.plot(xx, CubicSpline(x.detach(), y1.detach(), bc_type=bc_type)(xx.detach()))
    # plt.plot(x.detach(), y1.detach(), 'x')
    # plt.show()
    if bc_type == "periodic":
        rtol = 2e-2
    else:
        rtol = 1e-3
    assert torch.allclose(yq11, yq11_true, rtol=rtol)
    assert torch.allclose(yq12, yq12_true, rtol=rtol)
    assert torch.allclose(yq21, yq21_true, rtol=rtol)
    assert torch.allclose(yq22, yq22_true, rtol=rtol)

    # skip the gradient check if bc_type is None
    if bc_type is None:
        return

    gradcheck(interp, (x, y1, xq1))
    gradcheck(interp, (x, y1, xq2))
    gradcheck(interp, (x, y2, xq1))
    gradcheck(interp, (x, y2, xq2))

    gradgradcheck(interp, (x, y1, xq1))
    gradgradcheck(interp, (x, y1, xq2))
    gradgradcheck(interp, (x, y2, xq1))
    gradgradcheck(interp, (x, y2, xq2))

@device_dtype_float_test(only64=True, additional_kwargs={
    "scramble": [False, True]
})
def test_interp1_linear(dtype, device, scramble):
    dtype_device_kwargs = {"dtype": dtype, "device": device}
    x = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], **dtype_device_kwargs).requires_grad_()
    y1 = torch.tensor([1.0, 1.5, 2.1, 1.1, 2.3, 2.5], **dtype_device_kwargs).requires_grad_()
    y2 = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 2.5],
                       [0.8, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()

    # points are well inside to avoid extrapolation in numerical gradient calculations
    xq1 = torch.linspace(0.05, 0.95, 10, **dtype_device_kwargs)
    xq2 = torch.linspace(0.05, 0.95, 4, **dtype_device_kwargs)

    if scramble:
        idx1 = torch.randperm(len(xq1))
        idx2 = torch.randperm(len(xq2))
        xq1 = xq1[..., idx1]
        xq2 = xq2[..., idx2]
    xq1 = xq1.requires_grad_()
    xq2 = xq2.requires_grad_()

    # # true results (obtained from scipy.interpolate.interp1d)
    # from scipy.interpolate import interp1d
    # print("yq11:", interp1d(x.detach(), y1.detach())(xq1.detach()))
    # print("yq12:", interp1d(x.detach(), y1.detach())(xq2.detach()))
    # print("yq21:", interp1d(x.detach(), y2[1].detach())(xq1.detach()))
    # print("yq22:", interp1d(x.detach(), y2[1].detach())(xq2.detach()))

    yq11_true = torch.tensor([1.125, 1.375, 1.8, 1.85, 1.35, 1.3, 1.7, 2.1, 2.35, 2.45],
                             **dtype_device_kwargs)
    yq12_true = torch.tensor([1.125, 1.85, 1.7, 2.45], **dtype_device_kwargs)
    yq21_true = torch.tensor([[1.125, 1.375, 1.8, 1.85, 1.35, 1.3, 1.7, 2.1, 2.35, 2.45],
                              [0.9, 1.1, 1.7, 1.75, 0.85, 0.86666667, 1.8, 2.73333333, 2.7, 1.7]],
                             **dtype_device_kwargs)
    yq22_true = torch.tensor([[1.125, 1.85, 1.7, 2.45],
                              [0.9, 1.75, 1.8, 1.7]],
                             **dtype_device_kwargs)

    if scramble:
        yq11_true = yq11_true[..., idx1]
        yq12_true = yq12_true[..., idx2]
        yq21_true = yq21_true[..., idx1]
        yq22_true = yq22_true[..., idx2]

    def interp(x, y, xq):
        return Interp1D(x, y, method="linear")(xq)

    yq11 = interp(x, y1, xq1)
    yq12 = interp(x, y1, xq2)
    yq21 = interp(x, y2, xq1)
    yq22 = interp(x, y2, xq2)
    # import matplotlib.pyplot as plt
    # from scipy.interpolate import interp1d
    # xx = torch.linspace(0, 1, 1000, **dtype_device_kwargs)
    # xx2 = torch.linspace(-1, 2, 1000, **dtype_device_kwargs)
    # plt.plot(xx2, interp(x, y1, xx2).detach().numpy())
    # plt.plot(xx, interp1d(x.detach(), y1.detach())(xx.detach()))
    # plt.plot(x.detach(), y1.detach(), 'x')
    # plt.show()
    assert torch.allclose(yq11, yq11_true)
    assert torch.allclose(yq12, yq12_true)
    assert torch.allclose(yq21, yq21_true)
    assert torch.allclose(yq22, yq22_true)

    gradcheck(interp, (x, y1, xq1))
    gradcheck(interp, (x, y1, xq2))
    gradcheck(interp, (x, y2, xq1))
    gradcheck(interp, (x, y2, xq2))

    gradgradcheck(interp, (x, y1, xq1))
    gradgradcheck(interp, (x, y1, xq2))
    gradgradcheck(interp, (x, y2, xq1))
    gradgradcheck(interp, (x, y2, xq2))

@device_dtype_float_test(only64=True, additional_kwargs={
    "method": ["cspline", "linear"]
})
def test_interp1_editable_module(dtype, device, method):
    dtype_device_kwargs = {"dtype": dtype, "device": device}
    x = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], **dtype_device_kwargs).requires_grad_()
    y = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 2.5],
                      [0.8, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()
    xq = torch.linspace(0, 1, 10, **dtype_device_kwargs).requires_grad_()

    cls1 = Interp1D(x, y, method=method)
    cls2 = Interp1D(x, method=method)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cls1.assertparams(cls1.__call__, xq)
        cls2.assertparams(cls2.__call__, xq, y)

@device_dtype_float_test(only64=True)
def test_extrap(dtype, device):
    dtype_device_kwargs = {"dtype": dtype, "device": device}
    x = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], **dtype_device_kwargs).requires_grad_()
    y1 = torch.tensor([[1.0, 2.1, 1.5, 1.1, 2.3, 2.5],
                       [0.0, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()
    xq1 = torch.tensor([0.0, 1. / 3, 2. / 3, 3. / 3, -1. / 3, -1.0, -4. / 3, 4. / 3,
                        6. / 3, 7. / 3, 9. / 3], **dtype_device_kwargs).requires_grad_()

    # true results (obtained from scipy.interpolate.CubicSpline)
    nan = float("nan")
    yq_nan_true = torch.tensor([
        [1., 1.3127193, 1.7445744, 2.5, nan, nan, nan, nan, nan, nan, nan],
        [0., 2.13368966, 1.82654566, 1.2, nan, nan, nan, nan, nan, nan, nan],
    ], **dtype_device_kwargs)
    yq_mir_true = torch.tensor([
        [1., 1.3127193, 1.7445744, 2.5, 1.3127193, 2.5, 1.7445744, 1.7445744, 1., 1.3127193, 2.5],
        [0., 2.13368966, 1.82654566, 1.2, 2.13368966, 1.2, 1.82654566, 1.82654566, 0., 2.13368966, 1.2],
    ], **dtype_device_kwargs)
    yq_bnd_true = torch.tensor([
        [1., 1.3127193, 1.7445744, 2.5, 1., 1., 1., 2.5, 2.5, 2.5, 2.5],
        [0., 2.13368966, 1.82654566, 1.2, 0., 0., 0., 1.2, 1.2, 1.2, 1.2],
    ], **dtype_device_kwargs)
    yq_1_true = torch.tensor([
        [1., 1.3127193, 1.7445744, 2.5, 1., 1., 1., 1., 1., 1., 1.],
        [0., 2.13368966, 1.82654566, 1.2, 1., 1., 1., 1., 1., 1., 1.],
    ], **dtype_device_kwargs)
    cal = lambda x: x * 2.
    yq_cal_true = torch.tensor([
        [1., 1.3127193, 1.7445744, 2.5, -2. / 3, -2., -8. / 3, 8. / 3, 12. / 3, 14. / 3, 18. / 3],
        [0., 2.13368966, 1.82654566, 1.2, -2. / 3, -2., -8. / 3, 8. / 3, 12. / 3, 14. / 3, 18. / 3],
    ], **dtype_device_kwargs)
    extraps = ["nan", "mirror", "bound", 1.0, cal]
    yq_trues = [yq_nan_true, yq_mir_true, yq_bnd_true, yq_1_true, yq_cal_true]

    def interp(x, y, xq, extrap):
        return Interp1D(x, y, extrap=extrap, method="cspline", bc_type="natural")(xq)

    for extrap, yq_true in zip(extraps, yq_trues):
        print("Extrap: %s" % extrap)
        yq = interp(x, y1, xq1, extrap=extrap)
        assert torch.allclose(yq, yq_true, equal_nan=True)
