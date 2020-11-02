Derivatives of :obj:`xitorch.linalg.symeig`
===========================================

Author: Muhammad Firmansyah Kasim (2020)

Problem
-------

The function :obj:`xitorch.linalg.symeig` decomposes a linear operator to its
:math:`k` smallest or largest eigenvectors and eigenvalues,

.. math::
    \mathbf{AX} = \mathbf{MXE}

where :math:`\mathbf{A}, \mathbf{M}` are symmetric :math:`n\times n` linear operators
that act as the inputs of the function.
The outputs: :math:`\mathbf{X}` is an :math:`n\times k` matrix containing the eigenvectors
on its column, and :math:`\mathbf{E}` is a :math:`k\times k` diagonal matrix
containing the corresponding eigenvalues.

The linear operators :math:`\mathbf{A}` and :math:`\mathbf{M}` have parameters
that their elements depend on, which are denoted by :math:`\theta_A` and
:math:`\theta_M`, respectively.
In this case, we only consider 1 parameters for each linear operator.
Extending it to multiple parameters for one linear operator can be done
trivially because the obtained expression will be similar to other parameters.

In this page, we will derive the expression for backward derivative (a.k.a.
the vector-Jacobian product) of the linear operators parameters:
:math:`\overline{\theta_A} \equiv \partial \mathcal{L}/\partial \theta_A` and
:math:`\overline{\theta_M} \equiv \partial \mathcal{L}/\partial \theta_M` as functions of
:math:`\mathbf{\overline{X}} \equiv \partial \mathcal{L}/\partial \mathbf{X}` and
:math:`\mathbf{\overline{E}} \equiv \partial \mathcal{L}/\partial \mathbf{E}` for a loss value
:math:`\mathcal{L}`.
One challenge is that we only have implicit linear operators :math:`\mathbf{A}`
and :math:`\mathbf{M}` where they are expressed by their matrix-vector
multiplication and right-multiplications without explicit representation
on their matrix elements.
Another challenge is that only :math:`k` eigenpairs are available, so
calculations involving full eigenvectors and eigenvalues cannot be used.

This derivation assumes the eigenvalues are all unique.
Cases with degenerate eigenvalues are treated differently.

Forward derivative of a single eigenpair
----------------------------------------

Let's start with the eigendecomposition expression for one eigenvector and
eigenvalue,

.. math::
    \mathbf{Ax} = \lambda \mathbf{Mx},
    :label: eq:eigdecomp-single

where the eigenvector is normalized,

.. math::
    \mathbf{x}^T\mathbf{Mx} = 1.
    :label: eq:normalized-eivec

Applying first order derivative to the equations above we obtain,

.. math::
    \mathbf{A'x} + \mathbf{A}\mathbf{x'} = \lambda' \mathbf{Mx} +
        \lambda \mathbf{M'x} + \lambda \mathbf{Mx'}
    :label: eq:fwdderiv-eq-raw

and

.. math::
    \mathbf{x}^T \mathbf{M'x} + 2\mathbf{x}^T \mathbf{Mx'} = 0.
    :label: eq:fwdderiv-eivecs

Applying :math:`\mathbf{x}^T` on both sides of equation :eq:`eq:fwdderiv-eq-raw`,
we obtain

.. math::
    \mathbf{x}^T\mathbf{A'x} + \mathbf{x}^T\mathbf{A}\mathbf{x'} =
      \lambda' \mathbf{x}^T\mathbf{Mx} +
      \lambda \mathbf{x}^T\mathbf{M'x} + \lambda \mathbf{x}^T\mathbf{Mx'}.
    :label: eq:fwdderiv-eq-xtranspose

Substituting :math:`\mathbf{x}^T\mathbf{Mx}` from equation
:eq:`eq:normalized-eivec` and :math:`\mathbf{x}^T\mathbf{A}` from the transposed
equation :eq:`eq:eigdecomp-single`, we get the derivative of the eigenvalue,

.. math::
    \lambda' = \mathbf{x}^T(\mathbf{A'} - \lambda\mathbf{M'})\mathbf{x}.
    :label: eq:fwdderiv-eival

To obtain the derivative of the eigenvector, we substitute
:eq:`eq:fwdderiv-eival` to :eq:`eq:fwdderiv-eq-raw` and rearrange it to
obtain,

.. math::
    (\mathbf{A} - \lambda \mathbf{M})\mathbf{x'} =
      -(\mathbf{I} - \mathbf{Mxx}^T)(\mathbf{A'} - \lambda \mathbf{M'})\mathbf{x}
    :label: eq:fwdderiv-eivec-before-solve

The matrix :math:`(\mathbf{A} - \lambda \mathbf{M})` is not a full rank matrix,
so when multiplied to :math:`\mathbf{x'}`, some of its component is lost.
To solve this, we split :math:`\mathbf{x'}` into 2 components, orthogonal
(:math:`\mathbf{x_M'}`) and parallel (:math:`\mathbf{x_{-M}'}`):

.. math::
    \mathbf{x'} = \mathbf{x_M'} + \mathbf{x_{-M}'}
    :label: eq:split-xderiv

where

.. math::
    \left(\mathbf{I} - \mathbf{xx}^T\mathbf{M}\right) \mathbf{x_M'} &= \mathbf{x_M'} \\
    \left(\mathbf{I} - \mathbf{xx}^T\mathbf{M}\right) \mathbf{x_{-M}'} &= \mathbf{0}.
    :label: eq:split-xderiv-properties

Simple arrangement of the equations above yields

.. math::
    \mathbf{xx}^T\mathbf{M}\mathbf{x_M'} &= \mathbf{0} \\
    \mathbf{x_{-M}'} &= \mathbf{xx}^T\mathbf{M}\mathbf{x_{-M}'}.
    :label: eq:split-xderiv-properties-2

Using the equations :eq:`eq:split-xderiv-properties-2` in equation
:eq:`eq:fwdderiv-eivecs` and :eq:`eq:fwdderiv-eivec-before-solve` produces

.. math::
    \mathbf{x}^T\mathbf{Mx_{-M}'} &= -\frac{1}{2}\mathbf{x}^T\mathbf{M'x} \\
    (\mathbf{A} - \lambda \mathbf{M})\mathbf{x_M'} &=
      -(\mathbf{I} - \mathbf{Mxx}^T)(\mathbf{A'} - \lambda \mathbf{M'})\mathbf{x}.
    :label: eq:two-eqs-two-components

Multiplying the first equation above with :math:`\mathbf{x}` and using the second
equation from :eq:`eq:split-xderiv-properties-2`, we obtain,

.. math::
    \mathbf{x_{-M}'} = -\frac{1}{2}\mathbf{xx}^T\mathbf{M'x}.
    :label: eq:fwdderiv-eivecs-par

Moving the matrix :math:`(\mathbf{A} - \lambda \mathbf{M})` on the second equation
of :eq:`eq:two-eqs-two-components` to the right hand side gives us

.. math::
    \mathbf{x_M'} = -(\mathbf{I} - \mathbf{xx}^T\mathbf{M})(\mathbf{A} - \lambda \mathbf{M})^{+}
      (\mathbf{I} - \mathbf{Mxx}^T)(\mathbf{A'} - \lambda \mathbf{M'})\mathbf{x},
    :label: eq:fwdderiv-eivecs-ortho

where the symbol :math:`\mathbf{C}^{+}` indicates the pseudo-inverse of the matrix.
The additional term :math:`(\mathbf{I} - \mathbf{xx}^T\mathbf{M})` is to make sure
the result is orthogonal.
The calculation of the pseudo-inverse can be obtained using standard linear equation
solver.

To summarize, the forward derivatives are given by

.. math::
    \lambda' &= \mathbf{x}^T(\mathbf{A'} - \lambda\mathbf{M'})\mathbf{x}. \\
    \mathbf{x'} &= -\frac{1}{2}\mathbf{xx}^T\mathbf{M'x} -
      (\mathbf{I} - \mathbf{xx}^T\mathbf{M})(\mathbf{A} - \lambda \mathbf{M})^{+}
      (\mathbf{I} - \mathbf{Mxx}^T)(\mathbf{A'} - \lambda \mathbf{M'})\mathbf{x}.

Backward derivative
-------------------
From the forward derivatives, it is relatively straightforward to get the
backward derivatives.
Using the relation

.. math::
    \mathbf{P'} = \mathbf{QR'S} \implies
    \mathbf{\overline{R}} = \mathbf{Q}^T\mathbf{\overline{P}}\mathbf{S}^T,

we get

.. math::
    \mathbf{\overline{A}} &= \mathbf{xx}^T \overline{\lambda} -
    (\mathbf{I} - \mathbf{xx}^T\mathbf{M})(\mathbf{A} - \lambda \mathbf{M})^{+}
    (\mathbf{I} - \mathbf{Mxx}^T)\mathbf{\overline{x}} \mathbf{x}^T \\
    \mathbf{\overline{M}} &= -\mathbf{xx}^T \lambda \overline{\lambda}
    -\frac{1}{2}\mathbf{xx}^T\mathbf{\overline{x}}\mathbf{x}^T +
    \lambda (\mathbf{I} - \mathbf{xx}^T\mathbf{M})(\mathbf{A} - \lambda \mathbf{M})^{+}
    (\mathbf{I} - \mathbf{Mxx}^T)\mathbf{\overline{x}} \mathbf{x}^T.

For cases with multiple eigenpairs, the contributions should be summed from
all eigenvalues and eigenvectors,

.. math::
    \mathbf{\overline{A}} &= \mathbf{X\overline{E}X}^T - \mathbf{\overline{Y}X}^T\\
    \mathbf{\overline{M}} &= \mathbf{XE\overline{E}X}^T -
        \frac{1}{2}\mathbf{X}(\mathbf{I}\circ\mathbf{X}^T\mathbf{\overline{X}})\mathbf{X}^T +
        \mathbf{\overline{Y}EX}^T.
    :label: eq:multi-contrib-bckderiv

where :math:`\circ` indicates element-wise multiplication and

.. math::
    \mathbf{\overline{Y}} &= \mathbf{\overline{V}} -
        \mathbf{U}\left(\mathbf{I}\circ\mathbf{U}^T\mathbf{M\overline{V}}\right) \\
    \mathbf{\overline{V}} &:
        \mathrm{solve}\ \mathbf{A\overline{V}} - \mathbf{M\overline{V}E} =
        \mathbf{\overline{U}} -
        \mathbf{MU} \left(\mathbf{I}\circ\mathbf{U}^T \mathbf{\overline{U}}\right).
    :label: eq:y-and-v

Given the gradient of each elements in the linear operator,
the gradient with respect to the parameters of :math:`\mathbf{A}` and
:math:`\mathbf{M}` are

.. math::
    \overline{\theta_A} &= \mathrm{tr}\left(\mathbf{\overline{A}}^T
        \frac{\partial \mathbf{A}}{\partial \theta_A}\right) \\
    \overline{\theta_M} &= \mathrm{tr}\left(\mathbf{\overline{M}}^T
        \frac{\partial \mathbf{M}}{\partial \theta_M}\right)

or more conveniently written as

.. math::
    \overline{\theta_A} &= \mathrm{tr}\left[(\mathbf{X\overline{E} - \overline{Y}})^T
        \frac{\partial (\mathbf{AX})}{\partial \theta_A}\right] \\
    \overline{\theta_M} &= \mathrm{tr}\left[
        \left(\mathbf{XE\overline{E}} - \frac{1}{2}\mathbf{X}(\mathbf{I}\circ\mathbf{X}^T\mathbf{\overline{X}}) +
        \mathbf{\overline{Y}E}\right)^T
        \frac{\partial (\mathbf{MX})}{\partial \theta_M}\right].

In PyTorch, the terms above can be calculated by propagating the gradient from
:math:`\mathbf{AX}` or :math:`\mathbf{MX}` with initial gradient given on the
left term, e.g. :math:`(\mathbf{X\overline{E}} - \mathbf{\overline{Y}})` for
:math:`\overline{\theta_A}`.
