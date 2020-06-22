PCA
===
PCA = Principal Component Analysis

The goal of PCA is to reduce redudndancy (collapse redundant features), or to encode examples using a new set of
features

A technique to project a high-dimensional data matrix into a lower dimension.

- First normalize old features to mean :math:`\bar{x} = 0`, variance 1 unit
- Uses a linear transformation:
    - new features are projections, :math:`u \cdot x` for unit length u
    - projecting onto vectors :math:`u_1.. u_k` gives k new features - break each x into a sum of k vectors :math:`\mu_1 .. \mu_k`
- low spread in certain directions implies redundancy
- find projection directions that preserve as much variance as possible
- e.g. multiple different 3s can be viewed as a prototype rotated and shifted by latent variables
    - digits lie in noisy low-dimensional subspace of all images

First View: Preserve Variance
-----------------------------


- project onto u, but preserve variation
- length of :math:`x` in direction :math:`u` is :math:`u \cdot x` when :math:`u` has length 1
- maximize variance of projected instance lengths
- variance of projections onto :math:`u` over N training examples is:

.. math::
    & \frac{1}{N} \sum_{n=1}^N (\mathbf{u}^T \mathbf{x}_n - \mathbf{u}^T \mathbf{\bar{x}})^2 \\
    = & \frac{1}{N} \sum_{n=1}^N (\mathbf{u}^T \mathbf{x}_n)^2 \\
    = & \frac{1}{N} \sum_{n=1}^N \mathbf{u}^T \mathbf{x}_n \cdot \mathbf{x}_n^T \mathbf{u} \\
    = & \mathbf{u}^T (\frac{1}{N} \sum_{n=1}^N  \mathbf{x}_n \mathbf{x}_n^T) \mathbf{u} \\
    = & \mathbf{u}^T \mathbf{S} \mathbf{u}

Where :math:`\mathbf{S}` is the data covariance matrix :math:`\mathbf{X}^T \mathbf{X}`. Note that the term :math:`u^T \bar{x}` is 0 if the data is centered.

- Constrain :math:`u^Tu = 1`, add lagrange multiplier (penalizes violation of constraint), and maximize:
    - :math:`\max u^T Su+\lambda (1-u^Tu)`
- take matrix derivatives wrt :math:`u` and set equal to 0:
    - :math:`Su=\lambda u`
    - so :math:`u` is an eigenvector of :math:`S`.

.. note::
    Remember if :math:`M\mathbf{v} = \lambda \mathbf{v}` then:

    - :math:`\lambda` is an eigenvalue of matrix M
    - :math:`\mathbf{v}` is an eigenvector
    - if :math:`\mathbf{v}` is an eigenvector, then so is :math:`c\mathbf{v}` (with same eigenvalue)
    - usually take length-1 eigenvectors
    - :math:`d*d` matrices have at most :math:`d` orthogonal eigenvectors

Since we want to maximize variance :math:`\mathbf{u}^T \mathbf{Su}`, and :math:`\mathbf{u}` is an eigenvector,

.. math::
    \mathbf{u}^T \mathbf{Su} = \mathbf{u}^T \lambda \mathbf{u} = \lambda \mathbf{u}^T  \mathbf{u} = \lambda

so the first principal component is the eigenvector with largest eigenvalue! The second is the next, and so on

- eigenvector with largest eigenvalue is first principle component
- find other principle components iteratively by considering directions perpendicular to previous ones
- first k principle components will be first k eigenvectors of :math:`S`
- k eigenvectors can be found in :math:`O(kD^2)` time for a :math:`D*D` matrix

Second View: Data Compression
-----------------------------
Let's start over and look at this new view.

- goal: find a compressed set of :math:`k < D` features that approximates data
- use set of :math:`k` projections onto orthogonal unit vectors :math:`\mathbf{u}_1 .. \mathbf{u}_k`, which form an ortho-normal basis for the subspace projected onto
- projection of an :math:`\mathbf{x}_n` is :math:`(\alpha_{n, 1}, \alpha_{n, 2}, ..., \alpha_{n, k})` in the u-coordinates in the original space
    - :math:`\tilde{\mathbf{x}} = \sum_{i=1}^k \alpha_{n,i}\mathbf{u}_i`

We want to minimize average projection distance

- for 1-dimensional case, projection :math:`\tilde{\mathbf{x}} = (\mathbf{x}^T_n \mathbf{u})\mathbf{u}`
- goal: :math:`\min_{\mathbf{u}} J = \sum_{n=1}^N ||\mathbf{x}_n -\tilde{\mathbf{x}}||^2`
- :math:`J` is roughly a measure of how much information is lost during this compression

Assuming centered data and :math:`\mathbf{u}` unit length:

.. math::
    J & = \sum_{n=1}^N ||\mathbf{x}_n -\tilde{\mathbf{x}}||^2 \\
    & = \sum_{n=1}^N \mathbf{x}_n^T \mathbf{x}_n - 2 \mathbf{x}_n^T (\mathbf{x}^T_n \mathbf{u})\mathbf{u} + (\mathbf{x}^T_n \mathbf{u})^2 \mathbf{u}^T \mathbf{u} \\
    & = \sum_{n=1}^N \mathbf{x}_n^T \mathbf{x}_n - 2 (\mathbf{u}^T \mathbf{x}_n ) \mathbf{x}_n^T \mathbf{u} + (\mathbf{u}^T \mathbf{x}_n) (\mathbf{x}^T_n \mathbf{u}) \\
    & = \sum_{n=1}^N \mathbf{x}_n^T \mathbf{x}_n - \sum_{n=1}^N \mathbf{u}^T \mathbf{x}_n \mathbf{x}^T_n \mathbf{u}

So minimizing J wrt **u** means maximizing :math:`\sum_{n=1}^N \mathbf{u}^T \mathbf{x}_n \mathbf{x}^T_n \mathbf{u} = N \mathbf{u}^T \mathbf{Su}`.

And this is the same objective function as earlier!

Uses
----

- compression: approximate data with fewer features
- visualization: compress to 2 or 3 dimensions
- preprocess features to remove redundancy - improve speed, simplify hypothesis
- possible noise reduction
- plot eigenvalues - gives idea of dimensionality data lies in
