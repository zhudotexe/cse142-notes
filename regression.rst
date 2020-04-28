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

Assuming a Gaussian distribution on :math:`p`.

So to maximize the likelihood of *f*, we're just minimizing the squared error!

Convexity and Gradients
-----------------------
We know that the squared error is convex - we want to find the bottom of the bowl, where the gradient is 0.

.. note::
    For review: minimizing :math:`f(x) = (x - 5)^2`

    - :math:`f'(x) = 2(x - 5)`
    - :math:`f'(x) = 0`
    - :math:`x = 5` is the minimizer

So, we can derive the closed form from the ML formula:

.. math::
    \nabla_w \ln L(f) & = \nabla_w (m \ln (\frac{1}{\sqrt{2 \pi} \sigma}) - \frac{1}{2 \sigma^2} \sum_{i=1}^m (y^{(i)} - \mathbf{w} \cdot \mathbf{x^{(i)}})^2 \\
    & = (\frac{1}{\sigma^2} \sum_{i=1}^m (y^{(i)} - \mathbf{w}^T \cdot \mathbf{x^{(i)}})\mathbf{x_i^T}) \\
    \mathbf{0}^T & = \sum_{i=1}^m y^{(i)} \mathbf{x_i^T} - \mathbf{w}^T \sum_{i=1}^m \mathbf{x}^{(i)} \mathbf{x_i^T} \\
    ... & \text{some matrix magic...} \\
    \mathbf{\omega}_{ML} & = (X^T X)^{-1}X^T \vec{y}

.. note::
    Some of the matrix magic:

    .. math::
        \sum_{i=1}^m y^{(i)} \mathbf{x}^{(i)} = X^T \vec{y} \\

We can also use **SGD** to learn instead of finding minimum directly:

Cycle through samples, taking a step in the negative gradient direction for each example.

.. math::
    \omega_{new} & = \omega_{old} - \eta \nabla Error(\mathbf{x}^{(i)}, y^{(i)}) \\
    & = \omega_{old} - \eta \nabla \frac{1}{2} (y^{(i)} - \omega_{old} \cdot \mathbf{x}^{(i)})^2 \\
    & = \omega_{old} + \eta (y^{(i)} - \omega_{old} \cdot \mathbf{x}^{(i)}) \mathbf{x}^{(i)}

In this formula, :math:`\eta` is the learning rate. This is the LMS algorithm.

Bias-Variance Decomposition
---------------------------
Suppose we have training instances :math:`\mathbf{x}^{(1)}..\mathbf{x}^{(m)}` and true target function :math:`f`

Let labels :math:`y^{(i)}` in sample be :math:`f(\mathbf{x}^{(i)}) + \epsilon ^{(i)}` where :math:`\epsilon ^{(i)}` are
any iid noise.

Assume :math:`E[\epsilon ^{(i)}] = 0`, so :math:`E[y ^{(i)}] = f(\mathbf{x} ^{(i)})`

Let's examine the expected squared error between regression function :math:`g` learned from sample and "true"
function :math:`f` at a particular test point :math:`\mathbf{x}`

.. math::
    E_{noise} [(g(\mathbf{x}) - f(\mathbf{x}))^2]

where :math:`E_{noise}` is the expectation over training label noise.

Let :math:`\bar{g}(\mathbf{x})` be :math:`E_{noise} [(g(\mathbf{x})]`. We can rewrite
:math:`E_{noise} [(g(\mathbf{x}) - f(\mathbf{x}))^2]` as

.. math::
    & = E_{noise} [(g(\mathbf{x}) - \bar{g} + \bar{g} - f(\mathbf{x}))^2] \\
    & = E_{noise} [(g(\mathbf{x}) - \bar{g})^2 + (\bar{g} - f(\mathbf{x}))^2 + 2(g(\mathbf{x}) - \bar{g})(\bar{g} - f(\mathbf{x}))] \\
    & = E_{noise} [(g(\mathbf{x}) - \bar{g})^2] + E_{noise} [(\bar{g} - f(\mathbf{x}))^2] + 2E_{noise} [(g(\mathbf{x}) - \bar{g})(\bar{g} - f(\mathbf{x}))] \\
    & = E_{noise} [(g(\mathbf{x}) - \bar{g})^2] + (\bar{g} - f(\mathbf{x}))^2 + 2E_{noise} [(g(\mathbf{x}) - \bar{g})(\bar{g} - f(\mathbf{x}))] \\
    & = \text{variance of } g(\mathbf{x}) + (\text{bias of } g(\mathbf{x}))^2 + 2E_{noise} [(g(\mathbf{x}) - \bar{g})(\bar{g} - f(\mathbf{x}))]

Since :math:`f(\mathbf{x})` and :math:`\bar{g}` are constant with respect to noise,

.. math::
    E_{noise} [(g(\mathbf{x}) - \bar{g})(\bar{g} - f(\mathbf{x}))] & = (\bar{g} - f(\mathbf{x})) E_{noise}[(g(\mathbf{x}) - \bar{g})] \\
    & = (\bar{g} - f(\mathbf{x})) (E_{noise}[g(\mathbf{x})] - E_{noise}[\bar{g}]) \\
    & = (\bar{g} - f(\mathbf{x})) (\bar{g} - \bar{g}) \\
    & = 0.

ergo,

.. math::
    E_{noise} [(g(\mathbf{x}) - f(\mathbf{x}))^2] = \text{variance of } g(\mathbf{x}) + (\text{bias of } g(\mathbf{x}))^2

Regularized Least Squares
-------------------------

- Regularization penalizes complexity and reduces variance (but increases bias)
- adds a term :math:`\lambda` to the squared error (the regularization coefficient)

.. math::
    \sum_{i=1}^m (y^{(i)} - \mathbf{w}^T \mathbf{x}^{(i)})^2 + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w}

which is minimized by

.. math::
    \mathbf{w} = (\lambda \mathbf{I} + X^T X)^{-1} X^T \vec{y}

More generally, we can use a different regularizer:

.. math::
    \sum_{i=1}^n (y^{(i)} - \mathbf{w}^T \mathbf{x}^{(i)})^2 + \frac{\lambda}{2} \sum_{j=1}^m |w_j|^q

Logistic Regression
-------------------

Logistic Regression learns a parameter vector :math:`\mathbf{\theta}` (or weights :math:`\mathbf{w}`)

Idea behind 2-class logistic regression:

- two labels, so :math:`y \in \{0, 1\}` (binary classification)
- discriminative model gives :math:`p(y = 1 | \mathbf{x}; \mathbf{\theta})`
- labels softly separated by hyperplane
- maximum confusion at hyperplane: when :math:`\theta^T \mathbf{x} = 0` then :math:`p(y=1| \mathbf{x}; \theta) = 1/2`
- use add a dimension trick to shift hyperplane off :math:`\mathbf{0}`
- assume :math:`p(y = 1| \mathbf{x}; \theta)` is some :math:`g(\theta \cdot \mathbf{x})`

Properties of :math:`g(\theta \cdot \mathbf{x}) = g(\theta^T \mathbf{x}) = p(y = 1| \mathbf{x}; \theta)`:

- :math:`g(- \inf) = 0`
- :math:`g(\inf) = 1`
- :math:`g(0) = 1/2`
- confidence of label increases as you move away from the boundary - :math:`g(\theta^T \mathbf{x})` is monotonically increasing
- :math:`g(-a) = 1 - g(a)` (symmetry)

For example, one function that satisfies these properties is the logistic sigmoid.

.. math::
    h_\theta(\mathbf{x}) = g(\theta \cdot \mathbf{x}) = \frac{1}{1 + \exp(-\theta \cdot \mathbf{x})} = \frac{\exp(\theta \cdot \mathbf{x})}{1 + \exp(\theta \cdot \mathbf{x})}

And :math:`g()` has a simple derivative: :math:`g'(a) = g(a)(1-g(a))`.

Likelihood
^^^^^^^^^^

.. math::
    L(\theta) & = p(\vec{y} | X; \theta) \\
    & = \prod_{i = 1}^m p(y ^{(i)} | \mathbf{x} ^{(i)}; \theta) \\
    & = \prod_{i = 1}^m p(y = 1 | \mathbf{x} ^{(i)}; \theta) ^{y ^{(i)}} p(y = 0 | \mathbf{x} ^{(i)}; \theta) ^{1 - y ^{(i)}} \\
    & \text{^ this encodes the if-test on y} \\
    & = \prod_{i=1}^m h_\theta (\mathbf{x} ^{(i)}) ^{y ^{(i)}} (1 - h_\theta (\mathbf{x} ^{(i)})) ^{1-y ^{(i)}}

this is convex!

As before, log-likelihood is easier:

.. math::
    l(\theta) & = \log(L(\theta)) \\
    & = \sum_{i=1}^m y ^{(i)} \log (h_\theta (\mathbf{x} ^{(i)})) + (1-y ^{(i)}) \log (1 - h_\theta (\mathbf{x} ^{(i)}))

Taking the derivatives for one sample on one dimension of :math:`\theta`,

.. math::
    \frac{\partial}{\partial \theta_j} l(\theta) = (y ^{(i)} - h_\theta (\mathbf{x} ^{(i)})) x_j ^{(i)}

so we can then plug that, generalized into all dimensions, into SGD

.. math::
    \theta := \theta + \alpha (y ^{(i)} - h_\theta (\mathbf{x} ^{(i)})) \mathbf{x} ^{(i)}

Looks pretty similar to LMS, but :math:`h_\theta()` replaces :math:`\theta^T \mathbf{x}`.

Multiclass
^^^^^^^^^^
We can extend logistic regression to multiple classes:

- learn weights :math:`\theta_k` for each class :math:`k \in \{1..K\}`
- class-k-ness of instance :math:`\mathbf{x}` is estimated by :math:`\theta_k \cdot \mathbf{x}`
- estimate :math:`p(Class = k | \mathbf{x}; \theta_1 .. \theta_K)` for instance :math:`\mathbf{x}` using softmax fcn:

.. math::
    h_k(\mathbf{x}; \theta_1 .. \theta_K) = \frac{\exp(\theta_k \cdot \mathbf{x})}{\sum_{r=1}^K \exp(\theta_r \cdot \mathbf{x})}

- want weights that maximize likelihood of the sample
- use one-of-K encoding for labels
    - make each label :math:`\mathbf{y} ^{(i)}` a K-vector with :math:`y ^{(i)}_k = 1` iff class = k
- likelihood of :math:`m` labels in sample is:

.. math::
    L(\theta_1 .. \theta_K) & = p(\mathbf{y} ^{(1)} .. \mathbf{y} ^{(m)} | X; \theta_1 .. \theta_K) \\
    & = \prod_{i=1}^m p(\mathbf{y} ^{(i)} | \mathbf{x} ^{(i)}; \theta_1 .. \theta_K) \\
    & = \prod_{i=1}^m \prod_{k=1}^K h_k(\mathbf{x} ^{(i)}; \theta_1 .. \theta_K) ^{y_k ^{(i)}}

- iterative methods maximize log likelihood

.. note::
    Class :math:`\theta_K` is actually redundant, since :math:`p(class = K | \mathbf{x}) = 1 - \sum_{k=1}^{K-1} p(class = k | \mathbf{x})`.



