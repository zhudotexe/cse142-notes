Introduction
============

**Supervised learning:**

- Classification (binary/not)
- regression: continuous numeric labels
- ranking

**Batch assumption: iid**

- Distribution of things and measurements defines some unknown :math:`P(\mathbf{x}, y)` or :math:`D(\mathbf{x}, y)` over domain-label pairs
- We want to find a hypothesis :math:`f(\mathbf{x})` that is close to the truth
    - so we need a loss fcn :math:`L(y, t)` where y = prediction, t = truth
    - We want to minimize :math:`\int P(\mathbf{x},y) L(y, f(\mathbf{x}))`

**Overfitting and Underfitting**

*Overfitting* is when your hypothesis is too complex for the data, and

*Underfitting* is when your hypothesis is too simple.

Main Steps
----------

Feature Extraction
^^^^^^^^^^^^^^^^^^

In feature extraction, you need to extract some useful data dimension from your sample.
It's very important to extract good features from your data - there's a whole field around it (feature engineering).

Most commonly, an *instance* is represented as a vector of features :math:`\mathbf{x}`, along
with a ground-truth label :math:`y`. This can be represented as :math:`(\mathbf{x}, y)`.

They can be categorical/nominal, ordinal, or numeric.

**Example**

For example, in a binary spam/ham email classifier, some features you might extract might be:

- are there words in all caps?
- is the email long?
    - this can be represented as a binary attribute (after a certain length), or a numeric one.
- how many ``!`` are in the email?

Training
^^^^^^^^
In the training step, you choose a hypothesis space/hypothesis class and learn a *hypothesis*: a function that
maps an instance to a class/label.

The hypothesis :math:`g` is a learned model defined by parameters. We use ML when :math:`g(\mathbf{x})` is unknown to us
and we can't think about how to implement it algorithmically.

Testing
^^^^^^^
Now, given a new instance, we use our classifier to predict a label. Did it predict it correctly?

.. math::

    y' = g(\mathbf{x'})

It's important for classifiers to be able to *generalize*: to predict correctly when encountering a situation
that has never been seen before.

The power for a classifier to generalize often depends just as much on how you're representing instances as it does
the classification algorithm.

Generally, test data and training data should be drawn from the *same population*.

.. note::
    Independent, Identically Distributed (iid) assumption for same population:

    We assume the distribution of instances and labels defines some unknown, but fixed, :math:`P(\mathbf{x}, y)`.

    We also assume that all training and all test instances/labels are independent and identically distributed.

    That is - a collection of random variables is *iid* if they all have the same probability distribution,
    and are all mutually independent.

Supervised Learning
-------------------
Supervised learning is the primary type of ML. In this approach, training instances come with a ground-truth label.

Classification
^^^^^^^^^^^^^^
In a classification problem, labels are nominal (an unordered set). The labels can be either binary, or multiclass
(for example, a spam/ham classifier or a MNIST digit classifier).

Regression
^^^^^^^^^^
In a regression problem, labels are numeric (and continuous). For example, predicting the price of a house might
be a regression problem.

Ranking
^^^^^^^
In a ranking problem, the model is asked to order a set of objects. Usually, in this case, the input to these models
are keywords and a prior (e.g. prior data gathered on you), and the output is the objects' ranking.

More
^^^^
There's a lot more examples, too!

- Disease diagnosis
    - x: patient properties
    - y: disease/recommended therapy
- Part-of-speech tagging
    - x: an english sentence
    - y: the part of speech of a word in there
- Face recognition
    - x: bitmap of a person's face
    - y: identity of a person

- Reinforcement Learning
    - output: a sequence of actions (policy). Individual actions aren't important, but the overall policy is.
    - no supervised output, but delayed rewards
    - e.g. game playing, robot in a maze
- Online Learning
    - train on one instance at a time, as opposed to batch learning
    - e.g. perceptron

Other Forms
-----------

- unsupervised learning: no labels provided during train time
    - clustering
    - e.g. image compression, bioinformatics

- semi-supervised learning: use partially labeled data as well as unlabeled data

Training
--------
But what does training mean?

The hypothesis, :math:`g(\mathbf{x})`, is defined by parameters. How do we optimize these parameters?

We need to use a *loss function*.

Loss Function
^^^^^^^^^^^^^
To learn, we want to minimize the loss function :math:`L(y, y')`. This function measures the error of the prediction
:math:`y'` on the train set, and tells you how good :math:`g(\mathbf{x})` is at this point.

Choosing the right loss function is important, and different functions will want different functions.

.. note::
    For example, a simple loss function for binary classification is as follows:

    .. code-block:: python

        def L(y, y_pred):
            if y_pred == y:
                return 1
            else:
                return 0

Minimization is done using an optimization algorithm like gradient descent.

Stochastic Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^
aka SGD

In a sense, a function that, at an arbitrary point, finds the steepest slope that moves you in the direction of
the local minimum.

.. math::

    \text{Given:} \\
    t = [1..] \\
    \text{samples } (\mathbf{x}_t, y_t) \\
    \text{current model } f_t \\
    \text{learning rate } \eta_t \\ \\

    f_{t+1} = f_t - \eta_t \cdot \nabla_{f_t} L(f(\mathbf{x}_t), y_t)

Choices
^^^^^^^
**Model**

So really, our goal is to learn a *generalizable* :math:`g(x)` that is a **good approximation** of the truth.
You should choose a model that is capable of approximating the truth so.

**Hypothesis Space**

If we choose a hypothesis space that is too simple, there are fewer parameters to learn, but the model is less powerful.

The model will have less *variance* - fewer changes with changing training data - but more *bias* (making more
assumptions).

Buuut, choosing a hypothesis space that is too complex may cause high variance. This is called the bias-variance
tradeoff.

**Variance**

Having super high variance means you can perfectly represent the training data - but this is overfitting!

Different training sets may produce wildly different hypotheses - this is what *high variance* is.

**Bias**

Similarly, we say a model has *high inductive bias* when the model makes a lot of assumptions about the data.

For example, when we choose to use a linear regression, that model has a high bias towards a linear relationship
between the features and the labels.

**Bias-Variance Tradeoff**

Overall, variance represents estimation error (limits due to data) and bias represents approximation error
(limits due to model family).

These concepts are closely linked to the concept of overfitting/underfitting.

Evaluation
----------
There are lots of ways to measure predictive performance. The most popular are:

- Accuracy and Error Rate
- Precision Recall and F-measure

Accuracy
^^^^^^^^
Useful in X vs. Y problems, where any classes are equally important.

.. math::
    accuracy = \frac{\text{# correct predictions}}{\text{# test instances}}

.. math::
    error = 1 - accuracy = \frac{\text{# incorrect predictions}}{\text{# test instances}}

Confusion Matrix
^^^^^^^^^^^^^^^^
An extension of accuracy - given a matrix of positive and negative instances, it is possible to make a matrix

.. code-block:: text

             Predicted
               Y   N
             +-------
    Actual Y | TP  FN       Where TP = True Positive, FN = False Negative,
           N | FP  TN       FP = False Positive, TN = True Negative

**Example**: Given the result table

.. code-block:: text

             Predicted
               Y    N
             +--------
    Actual Y | 100  5
           N | 10   50

    Total corpus size: 165
    Total predicted yes: 110
    Total predicted no: 55
    Actual yes: 105
    Actual no: 60

    Accuracy = (100 + 50) / 160 = 0.91
    Error = (5 + 10) / 160 = 0.09

Precision, Recall, F-measure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
What about "X vs. non-X" types of problems: spam v. not spam, relevant v. not relevant?

.. math::
    precision = \frac{\text{# relevant records retrieved}}{\text{total # of records retrieved}}

.. math::
    recall = \frac{\text{# relevant records received}}{\text{total # of relevant records}}

.. math::
    F measure = \frac{2 * P * R}{P + R}

So, going back to our example, but now we're predicting whether or not something is relevant...

.. code-block:: text

             Predicted
               Y    N
             +--------
    Actual Y | 100  5
           N | 10   50

    Accuracy = (100 + 50) / 160 = 0.91

    Precision = TP / (TP + FP) = 100 / 110 = 0.91
    Recall = TP / (TP + FN) = 100 / 105 = 0.95
    F1 score = (2 * 0.91 * 0.95) / (0.91 + 0.95) = 0.93

**TLDR**:

.. code-block:: text

             Predicted
               Y   N
             +-------
    Actual Y | TP  FN
           N | FP  TN

.. math::
    accuracy = \frac{TP + TN}{P + N}

.. math::
    precision = \frac{TP}{TP + FP}

.. math::
    recall = \frac{TP}{TP + FN}

Reporting Performance
^^^^^^^^^^^^^^^^^^^^^

- separate training and test data
    - never see test data in training
    - usually 80-20 split
- k-fold cross validation
    - why just split once?
    - run k iterations, in each iteration hold out a different test set
    - improves robustness of reported set
