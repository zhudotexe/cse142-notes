Regressions
===========

Given training data :math:`\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1..m}`, find a linear approximation to the ys.

I.e., find **w** (or :math:`\theta`) of weights/parameters such that :math:`\theta \cdot \mathbf{x} \approx y`.

Topics:

- Least Squares as Maximum Likelihood
- Finding ML weights
- bias-variance error decomposition
- basis functions for normalizing features
- 1-norm, 2-norm regularization
- bayesian linear regression

Least Square
------------
Example:

.. code-block:: text

    Data
    x0 x1 | y
    ------+--
    1  2  | 5
    1  3  | 7
    1  4  | 9

Note the add-a-dimension trick of adding :math:`\mathbf{x_0} = \vec{1}`.

By eye, :math:`\theta = (1, 2)` has zero error (:math:`\theta = \mathbf{w}`)

We can represent the data matrix X as an array of feature vectors:

.. math::
    X =
    \begin{bmatrix}
    1 & 2 \\
    1 & 3 \\
    1 & 4
    \end{bmatrix}
    =
    \begin{bmatrix}
    \mathbf{x}^{(1)^T} \\
    \mathbf{x}^{(2)^T} \\
    \mathbf{x}^{(3)^T}
    \end{bmatrix}

and the label vector :math:`\mathbf{y}` / :math:`\vec{y}`:

.. math::
    \vec{y} =
    \begin{bmatrix}
    5 \\
    7 \\
    9
    \end{bmatrix}

.. note::
    To simplify our notation, :math:`\mathbf{x}^{(i)} \cdot \theta = \mathbf{x}^{(i)^T} \cdot \theta`.

Criterion
^^^^^^^^^

.. math::
    J(\theta) = \frac{1}{2} \sum_{i=1}^3 (\mathbf{x}^{(i)} \cdot \theta - y^{(i)})^2

.. note::
    .. math::
        X \theta - \vec{y} =
        \begin{bmatrix}
        \mathbf{x}^{(1)} \cdot \theta \\
        \mathbf{x}^{(2)} \cdot \theta \\
        \mathbf{x}^{(3)} \cdot \theta
        \end{bmatrix}
        - \vec{y} =
        \begin{bmatrix}
        \mathbf{x}^{(1)} \cdot \theta - y^{(1)} \\
        \mathbf{x}^{(2)} \cdot \theta - y^{(2)} \\
        \mathbf{x}^{(3)} \cdot \theta - y^{(3)}
        \end{bmatrix}

    so we can simplify this with matrix operations.


.. math::
    J(\theta) = \frac{1}{2} (X \theta - \vec{y})^T (X \theta - \vec{y})

SGD
^^^

Consider SGD: take the example where :math:`\mathbf{x}^{(1)} = (1, 2)`, :math:`y^{(1)} = 5`, and :math:`\theta` is
initialized to :math:`(1, 1)`.

The squared-error on this example is :math:`((1 * \theta_0) + (2 * \theta_1) - 5)^2 = 4`, and its contribution to
:math:`J(\theta)` is 2 (squared-error div 2).

At :math:`\theta = (1, 1)`:

.. math::
    &  \frac{\partial}{\partial \theta_0} \frac{1}{2} ((1 * \theta_0) + (2 * \theta_1) - 5)^2 \\
    & = \frac{2}{2} ((1 * \theta_0) + (2 * \theta_1) - 5) * 1 \\
    & = -2

and similarly,

.. math::
    &  \frac{\partial J}{\partial \theta_1} = \frac{2}{2} ((1 * \theta_1) + (2 * \theta_1) - 5) * 2 \\
    & = -4

With step size :math:`\frac{1}{20}`, update :math:`\theta' = \theta = \frac{1}{20} \nabla_\theta(J)`

so :math:`\theta' = (1, 1) - (-0.1, -0.2) = (1.1, 1.2)`. If we keep doing this, eventually :math:`\theta` becomes
:math:`(1, 2)`.

Closed Form
^^^^^^^^^^^

However, there's a way to minimize :math:`\theta` without having to do SGD: the formula is

.. math::
    (X^T X)^{-1} X^T \vec{y}

**Example**

.. math::
    X^T X & =
    \begin{bmatrix}
    1 & 1 & 1 \\
    2 & 3 & 4
    \end{bmatrix}
    \begin{bmatrix}
    1 & 2 \\
    1 & 3 \\
    1 & 4
    \end{bmatrix}
    =
    \begin{bmatrix}
    3 & 9 \\
    9 & 29
    \end{bmatrix} \\

    (X^T X)^{-1} & =
    \begin{bmatrix}
    29/6 & -9/6 \\
    -9/6 & 3/6
    \end{bmatrix} \\

    (X^T X)^{-1} X^T \vec{y} & =
    \begin{bmatrix}
    29/6 & -9/6 \\
    -9/6 & 3/6
    \end{bmatrix}
    \begin{bmatrix}
    1 & 1 & 1 \\
    2 & 3 & 4
    \end{bmatrix}
    \begin{bmatrix}
    5 \\
    7 \\
    9
    \end{bmatrix}
    =
    \begin{bmatrix}
    1 \\
    2
    \end{bmatrix}

Maximum Likelihood
------------------
This is a more generic solution concept. Making some assumptions, we can derive the least-squared criterion:

- learn a function *f* in a given class of functions (not necessarily linear)
- have *m* examples :math:`\{(\mathbf{x}^{(i)}, y^{(i)})\}` where :math:`y^{(i)} = f(\mathbf{x}^{(i)} + \epsilon^{(i)})`
    - where :math:`\epsilon^{(i)}` is some noise
- assume **x**'s are fixed, concentrate on *y*'s like a discriminative model
- assume :math:`\epsilon^{(i)}` s are iid draws from some mean 0 Gaussian distribution
- then, the probability of getting :math:`y^{(i)}` for :math:`\mathbf{x}^{(i)}` with *f* is:

.. math::
    p(y^{(i)} | \mathbf{x}^{(i)}, f) & = p(\epsilon^{(i)} = y^{(i)} - f(\mathbf{x}^{(i)})) \\
    & = \frac{1}{\sqrt{2 \pi} \sigma} \exp( -\frac{(y^{(i)} - f(\mathbf{x}^{(i)})^2)}{2 \sigma^2} )

with this, we can check the overall likelihood of datapoints over the entire set:

.. math::
    L(f) = P(\text{all labels } | f \text{, all } \mathbf{x}) = \prod_{i=1}^m p(y^{(i)} | \mathbf{x}^{(i)}, f)

using some log properties, we can...

.. math::
    \ln L(f) & = \sum_{i=1}^m \ln p(y^{(i)} | \mathbf{x}^{(i)}, f) \\
    \ln L(f) & = m \ln (\frac{1}{\sqrt{2 \pi} \sigma}) - \frac{1}{2 \sigma^2} \sum_{i=1}^m (y^{(i)} - f(\mathbf{x}^{(i)})^2

So to maximize the likelihood of *f*, we're just minimizing the squared error!
