Instance-Based Learning
=======================
aka nearest neighbor methods, non-parametric, lazy, memory-based, or case-based learning

In instance-based learning, there is no parametric model to fit
e.g. K-NN, some density estimation, locally weighted linear regression

Nearest Neighbor
----------------

- instances :math:`\mathbf{x}` are vectors of real numbers
- store the *m* training examples :math:`(\mathbf{x} ^{(1)}, y ^{(1)}) .. (\mathbf{x} ^{(m)}, y ^{(m)})`
- to predict on new :math:`\mathbf{x}`, find stored :math:`\mathbf{x} ^{(i)}` closest to :math:`\mathbf{x}` and predict :math:`y ^{(i)}`
    - def'n *closest*: the sample with the minimized square distance
    - different metrics for distance can be used
- Voronoi diagram: can detail the decision boundaries in 2D space

Note: it's important to use the right distance metric! If different dimensions have different scales (e.g. a dimension
from 0-1 vs. a dimension from 1k-1M), the smaller feature can become irrelevant. Similarly, adding irrelevant features
is problematic, as are highly correlated attributes

.. note::
    Example. 

    Let :math:`x_1 \in [0, 1]` determine class: :math:`y = 1 \iff x_1 > 0.3`

    Consider predicting the datapoint :math:`(0, 0)` given the data:

    - :math:`(0.1, x_2)` labeled 0
    - :math:`(0.5, x'_2)` labeled 1
    - where :math:`x_2, x'_2` are random draws from :math:`[0, 1]`

    What is the probability of mistake?

    If :math:`(0.1 ^2 + x_2 ^2) > (0.5 ^2 + x\prime _2 ^2 )`, :math:`(0, 0)` will be misclassified

    therefore the probability is :math:`P((0.1 ^2 + x_2 ^2) > (0.5 ^2 + x\prime _2 ^2 ))`.

    (note: this formula may not have been copied correctly)

    .. math::
        & = \int_{x=0}^1 P((0.1 ^2 + x_2 ^2) > (0.5 ^2 + x\prime _2 ^2 ), x_2 = x) dx \\
        & = \int_{x=0}^1 P((0.1 ^2 + x_2 ^2) > (0.5 ^2 + x\prime _2 ^2 ) | x_2 = x) * f_{x_2}(x) dx \\
        & = \int_{x=0}^1 P((0.1 ^2 + x_2 ^2 - 0.5 ^2) > (x\prime _2 ^2 )) * 1 dx \\
        & = \int_{x=0}^1 P((x\prime _2  ) < \sqrt{x_2^2 - 0.24} | x_2 = x) dx \\
        & = \int_{x=0}^1 \sqrt{x - 0.24} dx \\
        & \approx 0.275

There are some tricks, though:

- normalize attributes (e.g. mean 0, var 1 gaussian distribution)
- use a "mutual information" component :math:`w_j` on the *j* th component
    - :math:`dist(x, x') = \sum_j w_j (x_j - x'_j)^2`
    - :math:`w_j = I(x_j, y)`
- Mahalanobis distance - a covariance matrix

**Curse of Dimensionality**

As the number of attributes goes up, so does the "volume" - you need exponenitally many more points to cover the
training space


K-d Trees
^^^^^^^^^

We can greatly speed up finding the nearest neighbor by organizing a tree

- like BST, but organized around dimensions
- each node tests a single dimension against the threshold (median)
- can use highest variance dimension or cycle through dimensions
- growing a good tree can be expensive

Noise
^^^^^
Noise causes a problem in NN - if the nearest neighbor is noisy, there will be a misprediction.

So how do we make it robust against noise?

K-Nearest Neighbors
-------------------
In K-NN, we find the closest K points and predict given their majority vote

Given the law of large numbers and infinite data points and k = infinity, this should theoretically always be correct.

Nonparametric Regression
------------------------

- sometimes called "smoothing models"
- emphasize nearby points, e.g.
    - predict nearest neighbor
    - predict with distance-weighted average of labels
    - predict with locally weighted linear regression
        - divide into *h* bins, linreg on each bin

.. note::
    Both for kNN and bins, choosing *k* and *h* are important - when they are small, there is little bias
    but high variance (undersmoothing). When they are large, there's a large bias but little variance (oversmoothing).
