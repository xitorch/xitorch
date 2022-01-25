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
    x = torch.tensor([[0.0, 0.2, 0.3, 0.5, 0.8, 1.0],
                      [0.0, 0.1, 0.2, 0.6, 0.7, 1.0]], **dtype_device_kwargs).requires_grad_()
    if bc_type != "periodic":
        y1 = torch.tensor([1.0, 1.5, 2.1, 1.1, 2.3, 2.5], **dtype_device_kwargs).requires_grad_()
        y2 = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 2.5],
                           [0.8, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()
    else:
        y1 = torch.tensor([1.0, 1.5, 2.1, 1.1, 2.3, 1.0], **dtype_device_kwargs).requires_grad_()
        y2 = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 1.0],
                           [0.8, 1.2, 2.2, 0.4, 3.2, 0.8]], **dtype_device_kwargs).requires_grad_()

    # points are well inside to avoid extrapolation in numerical gradient calculations
    xq1 = torch.linspace(0.05, 0.95, 10, **dtype_device_kwargs).expand((2, -1)).contiguous()
    xq2 = torch.linspace(0.05, 0.95, 4, **dtype_device_kwargs).expand((1, -1)).contiguous()

    scramble = scramble and bc_type != "periodic"
    if scramble:
        idx1 = torch.randperm(xq1.shape[-1])
        idx2 = torch.randperm(xq2.shape[-1])
        xq1 = xq1[..., idx1]
        xq2 = xq2[..., idx2]
    xq1 = xq1.requires_grad_()
    xq2 = xq2.requires_grad_()

    # true results (obtained from scipy.interpolate.CubicSpline)
    # from scipy.interpolate import CubicSpline
    # print("yq11:", CubicSpline(x[0].detach(), y1.detach(), bc_type=bc_type)(xq1[0].detach()))
    # print("yq11:", CubicSpline(x[1].detach(), y1.detach(), bc_type=bc_type)(xq1[1].detach()))
    # print("yq12:", CubicSpline(x[0].detach(), y1.detach(), bc_type=bc_type)(xq2[0].detach()))
    # print("yq12:", CubicSpline(x[1].detach(), y1.detach(), bc_type=bc_type)(xq2[0].detach()))
    # print("yq21:", CubicSpline(x[0].detach(), y2[0].detach(), bc_type=bc_type)(xq1[0].detach()))
    # print("yq21:", CubicSpline(x[1].detach(), y2[1].detach(), bc_type=bc_type)(xq1[1].detach()))
    # print("yq22:", CubicSpline(x[0].detach(), y2[0].detach(), bc_type=bc_type)(xq2[0].detach()))
    # print("yq22:", CubicSpline(x[1].detach(), y2[1].detach(), bc_type=bc_type)(xq2[0].detach()))

    # get the y_trues from scipy
    if bc_type == "clamped":
        yq11_true = torch.tensor([[1.01599131, 1.23547394, 1.85950467, 2.02868906, 1.37102567, 1.04108172,
                                   1.42061722, 2.04849297, 2.4435166, 2.5061722],
                                  [1.15458402, 1.86457988, 2.10061827, 1.60125466, 0.9583273, 0.80051997,
                                   1.6879346, 2.67117749, 2.78487169, 2.55645772]],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([[1.01599131, 2.02868906, 1.42061722, 2.5061722],
                                  [1.15458402, 1.60125466, 1.6879346, 2.55645772]], **dtype_device_kwargs)
        yq21_true = torch.tensor([[1.01599131, 1.23547394, 1.85950467, 2.02868906, 1.37102567, 1.04108172,
                                   1.42061722, 2.04849297, 2.4435166, 2.5061722],
                                  [0.88289528, 1.76052359, 2.16598976, 1.0603244, -0.26361496, -0.40752535,
                                   1.84590232, 3.7494006, 2.95335265, 1.4876579]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[1.01599131, 2.02868906, 1.42061722, 2.5061722],
                                  [0.88289528, 1.0603244, 1.84590232, 1.4876579]],
                                 **dtype_device_kwargs)
    elif bc_type == "not-a-knot" or bc_type is None:  # default choice
        yq11_true = torch.tensor([[0.66219741, 1.06231845, 1.8959342, 2.01058952, 1.36963168, 1.02084725,
                                   1.33918614, 1.97824847, 2.56027129, 2.70749165],
                                  [1.16715909, 1.85784091, 2.12116477, 1.68053977, 1.07286932, 0.87042614,
                                   1.61068182, 3.03068182, 4.06704545, 3.62159091]],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([[0.66219741, 2.01058952, 1.33918614, 2.70749165],
                                  [1.16715909, 1.68053977, 1.61068182, 3.62159091]], **dtype_device_kwargs)
        yq21_true = torch.tensor([[0.66219741, 1.06231845, 1.8959342, 2.01058952, 1.36963168, 1.02084725,
                                   1.33918614, 1.97824847, 2.56027129, 2.70749165],
                                  [0.76045455, 1.78954545, 2.18206676, 1.22034801, 0.02198153, -0.21564631,
                                   1.62377841, 4.79190341, 6.67571023, 4.58110795]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[0.66219741, 2.01058952, 1.33918614, 2.70749165],
                                  [0.76045455, 1.22034801, 1.62377841, 4.58110795]],
                                 **dtype_device_kwargs)
    elif bc_type == "natural":
        yq11_true = torch.tensor([[1.03045416, 1.24263582, 1.85784168, 2.03025785, 1.37277695, 1.03808008,
                                   1.41177844, 2.04167374, 2.45428693, 2.52449066],
                                  [1.22309994, 1.84320018, 2.12197718, 1.64260534, 0.99525959, 0.81569312,
                                   1.67539169, 2.72584853, 2.97806783, 2.71644906]],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([[1.03045416, 2.03025785, 1.41177844, 2.52449066],
                                  [1.22309994, 1.64260534, 1.67539169, 2.71644906]], **dtype_device_kwargs)
        yq21_true = torch.tensor([[1.03045416, 1.24263582, 1.85784168, 2.03025785, 1.37277695, 1.03808008,
                                   1.41177844, 2.04167374, 2.45428693, 2.52449066],
                                  [0.9080941, 1.75071769, 2.18532016, 1.1250146, -0.17565687, -0.35561989,
                                   1.78956063, 4.01070104, 3.88485062, 2.26135521]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[1.03045416, 2.03025785, 1.41177844, 2.52449066],
                                  [0.9080941, 1.1250146, 1.78956063, 2.26135521]],
                                 **dtype_device_kwargs)
    elif bc_type == "periodic":
        yq11_true = torch.tensor([[0.88184647, 1.16754002, 1.87806756, 1.99916778, 1.3241823, 1.13211374,
                                   1.69017244, 2.25696675, 2.09041608, 1.31247223],
                                  [1.15648189, 1.86547041, 2.09245252, 1.5646367, 0.90258944, 0.76558635,
                                   1.72706493, 2.48167575, 1.94605786, 1.14727743]],
                                 **dtype_device_kwargs)
        yq12_true = torch.tensor([[0.88184647, 1.99916778, 1.69017244, 1.31247223],
                                  [1.15648189, 1.5646367, 1.72706493, 1.14727743]], **dtype_device_kwargs)
        yq21_true = torch.tensor([[0.88184647, 1.16754002, 1.87806756, 1.99916778, 1.3241823, 1.13211374,
                                   1.69017244, 2.25696675, 2.09041608, 1.31247223],
                                  [0.80193242, 1.78517005, 2.14439598, 1.02718087, -0.28362964, -0.41073874,
                                   1.84429108, 3.7595664, 2.94839885, 1.2942101]],
                                 **dtype_device_kwargs)
        yq22_true = torch.tensor([[0.88184647, 1.99916778, 1.69017244, 1.31247223],
                                  [0.80193242, 1.02718087, 1.84429108, 1.2942101]],
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
    x = torch.tensor([[0.0, 0.2, 0.3, 0.5, 0.8, 1.0],
                      [0.0, 0.1, 0.2, 0.6, 0.7, 1.0]], **dtype_device_kwargs).requires_grad_()
    y1 = torch.tensor([1.0, 1.5, 2.1, 1.1, 2.3, 2.5], **dtype_device_kwargs).requires_grad_()
    y2 = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 2.5],
                       [0.8, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()

    # points are well inside to avoid extrapolation in numerical gradient calculations
    xq1 = torch.linspace(0.05, 0.95, 10, **dtype_device_kwargs).expand((2, -1)).contiguous()
    xq2 = torch.linspace(0.05, 0.95, 4, **dtype_device_kwargs).expand((1, -1)).contiguous()

    if scramble:
        idx1 = torch.randperm(xq1.shape[-1])
        idx2 = torch.randperm(xq2.shape[-1])
        xq1 = xq1[..., idx1]
        xq2 = xq2[..., idx2]
    xq1 = xq1.requires_grad_()
    xq2 = xq2.requires_grad_()

    # # true results (obtained from scipy.interpolate.interp1d)
    # from scipy.interpolate import interp1d
    # print("yq11:", interp1d(x[0].detach(), y1.detach())(xq1[0].detach()))
    # print("yq11:", interp1d(x[1].detach(), y1.detach())(xq1[1].detach()))
    # print("yq12:", interp1d(x[0].detach(), y1.detach())(xq2[0].detach()))
    # print("yq12:", interp1d(x[1].detach(), y1.detach())(xq2[0].detach()))
    # print("yq21:", interp1d(x[0].detach(), y2[0].detach())(xq1[0].detach()))
    # print("yq21:", interp1d(x[1].detach(), y2[1].detach())(xq1[1].detach()))
    # print("yq22:", interp1d(x[0].detach(), y2[0].detach())(xq2[0].detach()))
    # print("yq22:", interp1d(x[1].detach(), y2[1].detach())(xq2[0].detach()))

    yq11_true = torch.tensor([[1.125, 1.375, 1.8, 1.85, 1.35, 1.3, 1.7, 2.1, 2.35, 2.45],
                              [1.25, 1.8, 1.975, 1.725, 1.475, 1.225, 1.7, 2.33333333, 2.4, 2.46666667]],
                             **dtype_device_kwargs)
    yq12_true = torch.tensor([[1.125, 1.85, 1.7, 2.45],
                              [1.25, 1.725, 1.7, 2.46666667]], **dtype_device_kwargs)
    yq21_true = torch.tensor([[1.125, 1.375, 1.8, 1.85, 1.35, 1.3, 1.7, 2.1, 2.35, 2.45],
                              [1., 1.7, 1.975, 1.525, 1.075, 0.625, 1.8, 2.86666667, 2.2, 1.53333333]],
                             **dtype_device_kwargs)
    yq22_true = torch.tensor([[1.125, 1.85, 1.7, 2.45],
                              [1., 1.525, 1.8, 1.53333333]],
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

@device_dtype_float_test(only64=True)
def test_interp1_unsorted(dtype, device):
    dtype_device_kwargs = {"dtype": dtype, "device": device}
    x = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], **dtype_device_kwargs).requires_grad_()
    y1 = torch.tensor([1.0, 1.5, 2.1, 1.1, 2.3, 2.5], **dtype_device_kwargs).requires_grad_()
    y2 = torch.tensor([[1.0, 1.5, 2.1, 1.1, 2.3, 2.5],
                       [0.8, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()

    # points are well inside to avoid extrapolation in numerical gradient calculations
    xq1 = torch.linspace(0.05, 0.95, 10, **dtype_device_kwargs)
    xq2 = torch.linspace(0.05, 0.95, 4, **dtype_device_kwargs)

    def interp(x, y, xq):
        return Interp1D(x, y, method="linear")(xq)

    def interp2(x, y, xq):
        return Interp1D(x, method="linear")(xq, y)

    # calculate the interpolated value with sorted x
    yq11 = interp(x, y1, xq1)
    yq12 = interp(x, y1, xq2)
    yq21 = interp(x, y2, xq1)
    yq22 = interp(x, y2, xq2)

    # scramble x and y1 and y2
    idx1 = torch.randperm(len(x))
    x = x[..., idx1]
    y1 = y1[..., idx1]
    y2 = y2[..., idx1]

    # calculate the interpolated value with unsorted x
    yq11_u = interp(x, y1, xq1)
    yq12_u = interp(x, y1, xq2)
    yq21_u = interp(x, y2, xq1)
    yq22_u = interp(x, y2, xq2)
    yq11_u2 = interp2(x, y1, xq1)
    yq12_u2 = interp2(x, y1, xq2)
    yq21_u2 = interp2(x, y2, xq1)
    yq22_u2 = interp2(x, y2, xq2)

    assert torch.allclose(yq11, yq11_u)
    assert torch.allclose(yq12, yq12_u)
    assert torch.allclose(yq21, yq21_u)
    assert torch.allclose(yq22, yq22_u)
    assert torch.allclose(yq11, yq11_u2)
    assert torch.allclose(yq12, yq12_u2)
    assert torch.allclose(yq21, yq21_u2)
    assert torch.allclose(yq22, yq22_u2)

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
