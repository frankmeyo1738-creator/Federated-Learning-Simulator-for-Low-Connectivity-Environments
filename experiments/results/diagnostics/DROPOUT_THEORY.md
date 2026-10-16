# Correction 4: selection-aware, finite-horizon dropout theory

## Source contract

The inspected source is `src/core/server.py` (`select_clients`,
`_collect_updates`, `_train_and_transmit`), `src/network/impairment.py`
(`transmit`, `_apply_dropout`, `_apply_packet_loss`), `src/main.py` and
`src/analytics/tracker.py`. Their hashes are in `layer3_provenance.json`.

* Each round selects `min(N, max(1, int(N*fraction_fit)))` distinct IDs using
  `random.sample` over all IDs. Existing disconnections do not exclude IDs.
* Each selected client trains before its update reaches `transmit`. The
  impairment layer models update rejection, not suppression of local work.
* `transmit` first counts the attempt. `_apply_dropout` returns true if the
  current round is earlier than the stored expiry; that attempt is lost.
* Otherwise an expired entry is removed and a fresh Bernoulli trigger with
  probability p is drawn. A trigger rejects the current attempt and sets
  expiry to `round_num + D`. A trigger can occur only on a selected attempt.
* A trigger at r blocks selected attempts in r through r+D-1. The client is
  eligible at r+D but may immediately trigger again. An unselected client's
  clock still advances: an expired entry is cleared on its next selection.
* D=0 still rejects the triggering attempt, with no future persistence.
* Only if disconnection did not reject the update is independent packet loss
  checked. Only a surviving update receives analytical bandwidth accounting
  and sampled propagation delay. Neither delay introduces another loss cause.
* `_collect_updates` counts each `None` result as one `dropped_updates`.
  Thus dropped + received = selected, and the metric combines existing
  disconnections, fresh triggers and packet loss. It is not an offline
  fraction of the entire population and not a measured physical-packet count.

All 110 YAMLs and CSVs are checked. N=10, m=5 and T=20 for every run. Durations
are fixed integers, not random distributions:

| Profile | Trigger p | Fixed D | Packet loss l |
|---|---:|---:|---:|
| Baseline | 0 | 0 | 0 |
| Urban | .05 | 1 | .01 |
| Rural | .15 | 2 | .07 |
| Severe | .35 | 3 | .20 |

## Exact marginal expectation under the ideal random model

Let q=m/N, L=max(D-1,0), and A_t be the probability that a given client is
eligible immediately before the possible trigger in round t. Initially
A_1=1. Current-round uniform selection is independent of prior client states
under the ideal independent-draw model.

A trigger in round s has probability q*p*A_s. Triggers for one client within
the preceding L rounds are mutually exclusive because the first would block
the rest. Therefore, with an empty sum equal to zero:

    A_t = 1 - q*p * sum(A_(t-j), j=1,...,min(L,t-1)).

Conditional on selection, the three disjoint loss causes in round t have
probabilities:

    persistent disconnection: 1 - A_t
    new trigger:              A_t*p
    packet loss only:         A_t*(1-p)*l

Consequently:

    r_t = 1 - A_t*(1-p)*(1-l)
    expected observed rate over T rounds = (1/T)*sum(r_t).

There are exactly m*T attempted updates, so this is an exact expectation of
the recorded ratio under the stated model, not a ratio-of-expectations
approximation. `marginal_dropout_theory` evaluates it with Python `Fraction`,
treating the decimal YAML probabilities as rational design values.

The stationary/time-average expression is:

    A_stationary = 1 / (1 + q*p*L)
    r_stationary = 1 - (1-p)*(1-l) / (1 + q*p*L).

The old formula is the special case q=1, with a stationary rather than
20-round horizon. Its mistake for this experiment was assuming full
selection. It is not a generally invalid renewal formula. Degenerate chains
can be periodic; the stationary expression is not a claim that every
finite-time state probability converges monotonically.

## Exact finite-population probability enumeration

A marginal mean alone cannot justify a binomial uncertainty interval. Client
selection within a round is without replacement and losses persist in time.
`finite_dropout_distribution` therefore enumerates a Markov reward process.

At the start of a round let n=(n_0,...,n_L) count clients with 0,...,L rounds
of inherited blocking remaining. A selected-count vector s has probability:

    P(s | n) = product(comb(n_i,s_i)) / comb(N,m),  sum(s_i)=m.

This is multivariate hypergeometric selection from the actual finite N.
Conditional on selection:

    B ~ Binomial(s_0,p)       new triggers
    C ~ Binomial(s_0-B,l)     packet losses among remaining online updates
    lost = sum(s_i, i>0) + B + C.

All existing clocks decrement, regardless of selection. B previously
eligible clients enter state L for the next round. For L=0 they remain
eligible next round, while their current trigger attempts are still lost.

Starting at (N,0,...,0), the program propagates the joint distribution of
state counts and cumulative lost updates for 20 rounds. It sums possible
events, not random simulations. No outcome parameters are fitted to data.
Combinatorial enumeration is exact under the ideal model; PMF calculations
use float64, so small numerical rounding errors remain. Probability mass is
checked, never forcibly renormalized. The distribution mean must agree with
the independent rational marginal calculation within 1e-12.

## Validation independent of the population-state calculation

Targeted tests separately enumerate **labelled clients and absolute expiry
rounds**, using exact rational event probabilities on small populations.
They enumerate every selected subset, trigger branch and packet-loss branch
and compare the entire lost-count distribution with the population-state
algorithm. This uses a different state representation and transition code.

Additional tests exercise the unmodified production engine with synthetic
updates and controlled random outcomes: persistence, expiry after unselected
rounds, immediate re-triggering, duration zero, and packet-loss short
circuiting. No client training is executed. Boundary distributions, complete
probability mass, horizon dependence and p=0 / p=1 / l=0 / l=1 are checked.
The unchanged `select_clients` method is extracted from its source AST and
executed with a selection-call spy, without constructing a server or importing
plotting dependencies. It verifies that all IDs, including disconnected ones,
are passed to sampling and that the configured selected count is respected.
A further test changes only synthetic empirical comparison values and verifies
that predictions and prediction intervals stay unchanged.

## Uncertainty and empirical comparison

Use one preselected canonical stratum per profile: FedAvg IID, seeds 1-10.
Paired algorithm and partition strata repeat network seeds and are not
additional independent replications. Their observed counts remain reported.

The single-run PMF gives the predicted run SD. Convolving it ten times gives
the model distribution for the sum of ten independent runs; division by
1,000 attempts gives the distribution of the ten-seed mean drop percentage.
Report equal-tail 95% predictive intervals with actual discrete coverage.
These are **prediction intervals under fixed design parameters**, not
confidence intervals estimating p, l or a population mean from the data.

For transparency, also report exploratory Bonferroni bands: each profile
band targets 98.75% coverage, yielding at least 95% joint coverage across
four profile means by the union bound. Cross-profile independence is not
required for that bound. Ten-run convolution still assumes independent
seed runs. The wider bands must not be used to erase a pointwise discrepancy.

These probabilities idealize a deterministic finite-precision PRNG as
independent draws. They predict the experiment design, not exact outcomes of
seeds 1-10. The CSVs lack per-client selection and loss-cause traces, so they
cannot uniquely diagnose a residual discrepancy or verify historical event
ordering beyond the source/configuration evidence. No real-network model,
energy result or empirical calibration is inferred.

## Findings and wording boundaries

The deterministic predictions are 0%, 5.95%, 26.208599% and 60.732510%.
Observed rates are 0%, 5.60%, 30.20% and 59.80%. Rural is 3.991401 pp above
prediction and outside its pointwise 95% predictive interval [23.1,29.4]%,
although inside the exploratory four-profile band [22.2,30.3]%. Other
profiles are within their pointwise intervals. Do not claim universal
agreement or treat absence of a band exceedance as proof of correctness.

For Severe, changing full selection to q=.5 changes the stationary value
from 69.411765% to 61.481481%; the finite-start correction then gives
60.732510%. Thus the old gap cannot be attributed entirely to startup.

The final report and AI log are unchanged by this task. Section 5.6 should
replace the old formula and blanket agreement claim with this derivation,
uncertainty interpretation and explicitly retained Rural discrepancy.
