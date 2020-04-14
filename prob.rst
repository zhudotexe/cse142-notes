Probability Review
==================

Useful notes: http://cs229.stanford.edu/section/cs229-prob.pdf

Let's define some important things.

- **Outcome Space**: :math:`\Omega` - contains all possible atomic outcomes
- each outcome (atom) has probability *density* or *mass* (discrete v. continuous spaces)
- an *event* is a subset of :math:`\Omega`
- :math:`P(event)` is a sum (or integral) over event's outcomes
- a *random variable* :math:`V` maps :math:`\Omega` to (usually) :math:`R`
- :math:`V = value` is an event, :math:`P(V)` is a distribution

.. note::
    **Example**: rolling a fair 6-sided die and then flipping that many fair coins

    - :math:`\Omega = \{(1, H), (1, T), (2, HH), (2, HT), ...\}`
    - let number of heads be a random variable
    - so what's the expected number of heads?
        - :math:`E(V) = \sum_{\text{atoms } a} P(a)V(a)`

    Let's look at some other properties to figure this out.

- Events A and B are *independent* iff:
    - :math:`P(A \text{ and } B) = P(A) * P(B)`
- Conditional probability
    - :math:`P(A | B) = \frac{P(A, B)}{P(B)}`
- Product Rule
    - :math:`P(A, B) = P(A|B) * P(B)`
    - :math:`P(B, A) = P(B|A) * P(A)`
- **Bayes Rule**
    - :math:`P(A|B) = P(B|A) \frac{P(A)}{P(B)}`

- Expectations add
    - :math:`E(V_1 + V_2) = E(V_1) + E(V_2)`
- rule of conditioning (sum rule)
    - if events :math:`e_1, e_2, ... , e_k` partition :math:`\Theta` then:
    - :math:`P(event) = \sum P(e_i) P(event | e_i)`
    - :math:`E(rv) = \sum P(e_i) E(rv | e_i)`

.. note::
    Back to the expected number of heads.

    .. math::

        E(\text{# heads}) & = \sum_{r=1}^6 P(roll = r) E(\text{# heads} | roll = r) \\
        & = \frac{1}{6}(\frac{1+2+3+4+5+6}{2}) \\
        & = \frac{21}{12} = 1.75

- Joint distributions factor
    - if :math:`\Omega = (S*T*U)`, then :math:`P(S=s,T=t,U=u)` is :math:`P(S=s)P(T=t|S=s)P(U=u|S=s,T=t)`
- Conditional distributions are also distributions
    - :math:`P(A|B) = \frac{P(A, B)}{P(B)}`, so :math:`P(A|B, C)=\frac{P(A,B|C)}{P(B|C)}`

Bayes Rule for Learning
-----------------------

- Assume joint distribution :math:`P(\mathbf{X=x}, Y=y)`
- We want :math:`P(Y=y|\mathbf{X=x})` for each label :math:`y` on a new instance :math:`\mathbf{x}`
- So, using Bayes' Rule, :math:`P(y|\mathbf{x}) = P(\mathbf{x}|y) \frac{P(y)}{P(\mathbf{x})}`
- :math:`P(\mathbf{x})` doesn't matter here, so we care that :math:`P(y|\mathbf{x})` is proportional to :math:`P(\mathbf{x}|y) P(y)`
- From the data, we can learn :math:`P(\mathbf{x}|y)` and :math:`P(y)`
- Predict label :math:`y` with largest product

So how do we learn :math:`P(\mathbf{x}|y)`?

.. note::
    Take for example a coin flip. You observe the sequence HTH; what is the probability that the next flip is H?

    Mathematically, the answer is 2/3: taking the likelihood function :math:`L(\theta) = P(HTH|\theta)`
    we get the probability equal to :math:`\theta^2 (1-\theta)`.

    By finding the :math:`\theta` value at the zero derivative, we get 2/3.

.. note::
    But what if we have a prior belief :math:`P(\theta)` where :math:`\theta = P(H)`?

    Now, the posterior on :math:`\theta` becomes :math:`P(\theta | HTH)`:

    .. math::
        P(\theta | HTH) = P(HTH | \theta) \frac{P(\theta)}{P(HTH)}

    Or in this case:

    .. math::
        \frac{\theta^2 (1-\theta) P(\theta)}{normalization}

    **Discrete Prior**

    Taking :math:`P(\theta=0) = P(\theta=1/2) = P(\theta=1) = 1/3`, :math:`\theta^2 (1-\theta) P(\theta)` is
    0, 1/24, and 0 for the 3 cases respectively. Thus, the posterior :math:`P(\theta = 1/2 | HTH) = 1`.

    **Prior Density**

    - :math:`P(\theta) = 1` for :math:`0 \leq \theta \leq 1`
    - So :math:`\theta^2 (1-\theta) P(\theta)` is just :math:`\theta^2 (1-\theta)`
    - and the posterior is :math:`\frac{\theta^2 (1-\theta)}{12}`
    - If we plot this, the max is at :math:`\theta = 2/3`

- Treat parameter :math:`\theta` as a random var with the prior distribution :math:`P(\theta)`, see training data :math:`Z`
- :math:`posterior = \frac{prior * data likelihood}{constant}`
- :math:`P(\theta | Z) = \frac{P(\theta) P(Z | \theta)}{P(Z)}`

Bayes' Estimation
-----------------

Treat parameter :math:`\theta'` as a RV with the prior distribution :math:`P(\theta)`, use fixed data
:math:`Z = (\mathbf{x}, y)` (RV :math:`S`)

Maximum Likelihood
^^^^^^^^^^^^^^^^^^

.. math::
    \theta_{ML} = \arg \max_{\theta'} P(S=Z|\theta = \theta')

Maximum a Posteriori
^^^^^^^^^^^^^^^^^^^^

.. math::
    \theta_{MAP} & = \arg \max_{\theta'} P(\theta = \theta' | S=Z) \\
    & = \arg \max_{\theta'} P(S=Z | \theta = \theta')\frac{P(\theta = \theta')}{P(S=Z)}

Predictive Distribution
^^^^^^^^^^^^^^^^^^^^^^^
aka Full Bayes

.. math::
    P(Y=y | S=Z) = \int P(Y=y | \theta=\theta') P(\theta=\theta' | S=Z) d\theta'

Mean a'Post
^^^^^^^^^^^

.. math::
    \theta_{mean} = E[\theta | S=Z] = \int \theta' P(\theta=\theta' | S=Z) d\theta'

Use
^^^

- draw enough data so that :math:`P(Y=y | X=\mathbf{x})` estimated for every possible pair
    - this takes a lot of data
- another approach: class of models
- think of each model :math:`m` as a way of generating the training set Z of :math:`(\mathbf{x}, y)` pairs

Compound Experiment
^^^^^^^^^^^^^^^^^^^

- prior :math:`P(M=m)` on model space
- models give :math:`P(X=x | M=m)` (where :math:`x` is a pair :math:`(\mathbf{x}, y)`)
- The joint experiment (if data is iid given m) is:

.. math::
    P(\{(\mathbf{x_i}, y_i)\}, m) = P(m) \prod_i (P(\mathbf{x_i} | m) P(y_i | \mathbf{x_i}, m))

Generative and Discriminative Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Generative model: :math:`P((\mathbf{x}, y) | m)`
    - tells how to generate examples (both instance and label)
    - learn :math:`P(\mathbf{x} | y, m)` and use Bayes' rule
    - common assumptions:
        - :math:`P(\mathbf{x} | y, m)` is Gaussian
        - :math:`P(y | m)` is Bernoulli
- Discriminative model: :math:`P(y | h, \mathbf{x})`
    - tells how to create labels from instances
    - often :math:`f(\mathbf{x}) = \arg \max_y f_y(\mathbf{x})`



