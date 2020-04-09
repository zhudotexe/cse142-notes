Probability Review
==================

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