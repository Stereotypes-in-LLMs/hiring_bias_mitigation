"""Statistical inference for counterfactual audits.

Three tests, deliberately reported side by side:

`permutation_test`
    The unpaired test of An et al. (2024), reproduced exactly (K=5000, two-sided, alpha=0.05)
    so numbers here stay directly comparable with the audit literature.

`paired_permutation_test`
    The test this design actually licenses. Our counterfactual sets are perfectly matched --
    same CV, same job, one attribute changed -- so the attribute's effect can be measured
    within a pair and the between-pair variance removed. The audit paper (Section 7) records
    the unpaired test as a limitation and names this as the alternative. It is strictly more
    powerful here, and it is the test a mitigation claim should rest on: mitigation effects
    are smaller than the raw disparities they remove, so the extra power matters.

`benjamini_hochberg`
    False-discovery-rate control across the whole family of tests in a run. Reporting
    requirement 1 of the audit paper, which that study itself does not meet. A design of this
    shape runs (attributes x measures x models x languages x conditions) tests; at an
    uncorrected alpha = 0.05 a proportion of flags is expected under the null even from a
    perfectly fair model. Every report prints raw and corrected counts side by side.

`bootstrap_ci` supplies the effect-size interval that reporting requirement 2 asks for: a
fractional difference in embedding similarity and a 58.7-point difference in acceptance rate
must not be presented with equal visual weight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_PERMUTATIONS = 5000
ALPHA = 0.05


@dataclass
class TestResult:
    group_mean: float
    population_mean: float
    observed_diff: float
    p_value: float
    significant: bool
    n_group: int
    n_population: int
    test: str
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def permutation_test(
    group_values: np.ndarray,
    population_values: np.ndarray,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    alpha: float = ALPHA,
    seed: int | None = 42,
) -> TestResult:
    """Unpaired permutation test of a group mean against the population mean.

    Reproduces Algorithm 1 of the audit paper, including its quirk that the resampled subsets
    are drawn from the full sample, which contains the group itself. That makes the test
    slightly conservative; it is retained for comparability, not because it is optimal.
    """
    group_values = np.asarray(group_values, dtype=float)
    population_values = np.asarray(population_values, dtype=float)
    n = group_values.size
    if n == 0 or population_values.size == 0:
        return TestResult(
            float("nan"), float("nan"), float("nan"), float("nan"), False, n,
            population_values.size, "permutation",
        )

    pop_mean = float(population_values.mean())
    observed = float(group_values.mean()) - pop_mean

    rng = _rng(seed)
    # Vectorised: one (K, n) index matrix instead of a Python loop over K draws. Same
    # procedure, ~200x faster, which is what makes FDR control over thousands of tests cheap.
    idx = rng.integers(0, population_values.size, size=(n_permutations, n))
    permuted = population_values[idx].mean(axis=1) - pop_mean
    p = float(np.mean(np.abs(permuted) >= abs(observed)))

    lo, hi = bootstrap_ci(group_values, seed=seed)
    return TestResult(
        group_mean=float(group_values.mean()),
        population_mean=pop_mean,
        observed_diff=observed,
        p_value=p,
        significant=p < alpha,
        n_group=n,
        n_population=int(population_values.size),
        test="permutation",
        ci_low=lo,
        ci_high=hi,
    )


def paired_permutation_test(
    group_values: np.ndarray,
    matched_values: np.ndarray,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    alpha: float = ALPHA,
    seed: int | None = 42,
) -> TestResult:
    """Sign-flip permutation test on matched counterfactual differences.

    `group_values[i]` and `matched_values[i]` are the same job-CV pair under two attribute
    values (typically the attribute under test and the rest of its counterfactual set). Under
    the null that the attribute does not matter, the sign of each within-pair difference is
    exchangeable, so flipping signs at random generates the null distribution.
    """
    a = np.asarray(group_values, dtype=float)
    b = np.asarray(matched_values, dtype=float)
    if a.size != b.size:
        raise ValueError(f"paired test needs equal lengths, got {a.size} and {b.size}")
    mask = ~(np.isnan(a) | np.isnan(b))
    diffs = a[mask] - b[mask]
    n = diffs.size
    if n == 0:
        return TestResult(
            float("nan"), float("nan"), float("nan"), float("nan"), False, 0, 0, "paired"
        )

    observed = float(diffs.mean())
    rng = _rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    permuted = (signs * diffs).mean(axis=1)
    p = float(np.mean(np.abs(permuted) >= abs(observed)))

    lo, hi = bootstrap_ci(diffs, seed=seed)
    return TestResult(
        group_mean=float(a[mask].mean()),
        population_mean=float(b[mask].mean()),
        observed_diff=observed,
        p_value=p,
        significant=p < alpha,
        n_group=n,
        n_population=n,
        test="paired_permutation",
        ci_low=lo,
        ci_high=hi,
    )


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = ALPHA,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean. Reporting requirement 2 (effect sizes)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = _rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def benjamini_hochberg(
    p_values: list[float], alpha: float = ALPHA
) -> tuple[list[bool], list[float]]:
    """BH step-up FDR control. Returns (rejected, adjusted p-values), input order preserved.

    NaN p-values (a test that could not run, e.g. an attribute with no usable outputs) are
    carried through as not-rejected rather than dropped, so the returned lists always line up
    with the attribute rows that produced them.
    """
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    rejected = np.zeros(p.size, dtype=bool)
    adjusted = np.full(p.size, np.nan)

    m = int(valid.sum())
    if m == 0:
        return rejected.tolist(), adjusted.tolist()

    idx = np.flatnonzero(valid)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranked = p[order]
    # Step-up: adjusted_(i) = min over j>=i of (m/j) * p_(j), enforced monotone from the top.
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    adjusted[order] = adj
    rejected[order] = adj <= alpha
    return rejected.tolist(), adjusted.tolist()


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions (acceptance rates).

    A percentage-point gap is the number a reader cares about, but it is not comparable
    across models with very different base rates -- 5 points at a 50% base rate is a much
    smaller effect than 5 points at a 5% base rate. h is reported next to the gap for that
    reason. Conventional reading: 0.2 small, 0.5 medium, 0.8 large.
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        return float("nan")
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
