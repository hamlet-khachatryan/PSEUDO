import numpy as np
import pytest
from scipy.stats import t as scipy_t

from quantify.statistical_model import (
    compute_significance_threshold,
    estimate_null_fraction,
    fit_null_distribution,
    fit_null_truncated_mle,
    fit_t_test,
)

TRUE_DF = 8.0
TRUE_LOC = 0.0
TRUE_SCALE = 1.0
RNG_SEED = 42


def _clean_samples(n=20000, seed=RNG_SEED):
    return scipy_t.rvs(df=TRUE_DF, loc=TRUE_LOC, scale=TRUE_SCALE, size=n, random_state=seed)


def _contaminated_samples(n=20000, frac_signal=0.10, signal_loc=3.0, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_signal = int(n * frac_signal)
    n_null = n - n_signal
    null_part = scipy_t.rvs(df=TRUE_DF, loc=TRUE_LOC, scale=TRUE_SCALE, size=n_null, random_state=seed)
    sig_part = scipy_t.rvs(df=TRUE_DF, loc=signal_loc, scale=TRUE_SCALE, size=n_signal, random_state=seed + 1)
    return rng.permutation(np.concatenate([null_part, sig_part]))

def test_fit_null_distribution_returns_dict_keys():
    samples = np.random.normal(loc=0.5, scale=1.2, size=500)
    params = fit_null_distribution(samples)
    assert set(params.keys()) == {"df", "loc", "scale"}


def test_fit_null_distribution_values_are_finite():
    samples = np.random.normal(size=300)
    params = fit_null_distribution(samples)
    assert all(np.isfinite(v) for v in params.values())


def test_fit_null_distribution_empty_input():
    params = fit_null_distribution(np.array([]))
    assert params == {"df": 1.0, "loc": 0.0, "scale": 1.0}


def test_fit_null_distribution_scale_positive():
    samples = np.random.normal(size=200)
    params = fit_null_distribution(samples)
    assert params["scale"] > 0


def test_compute_significance_threshold_returns_float():
    params = {"df": 10.0, "loc": 0.0, "scale": 1.0}
    result = compute_significance_threshold(params, alpha=0.05)
    assert isinstance(result, float)


def test_compute_significance_threshold_decreases_with_alpha():
    params = {"df": 10.0, "loc": 0.0, "scale": 1.0}
    t_strict = compute_significance_threshold(params, alpha=0.01)
    t_lenient = compute_significance_threshold(params, alpha=0.10)
    assert t_strict > t_lenient


def test_compute_significance_threshold_matches_scipy():
    params = {"df": 5.0, "loc": 1.0, "scale": 2.0}
    result = compute_significance_threshold(params, alpha=0.05)
    expected = scipy_t.ppf(0.95, df=5.0, loc=1.0, scale=2.0)
    assert abs(result - expected) < 1e-9

def test_fit_t_test_shape_preserved():
    params = {"df": 10.0, "loc": 0.0, "scale": 1.0}
    snr_map = np.random.normal(size=(4, 5, 6)).astype(np.float32)
    p_values = fit_t_test(params, snr_map)
    assert p_values.shape == snr_map.shape


def test_fit_t_test_values_in_unit_interval():
    params = {"df": 10.0, "loc": 0.0, "scale": 1.0}
    snr_map = np.random.normal(size=(3, 3, 3)).astype(np.float32)
    p_values = fit_t_test(params, snr_map)
    assert np.all(p_values >= 0.0)
    assert np.all(p_values <= 1.0)


def test_fit_t_test_high_snr_gives_low_pvalue():
    params = {"df": 10.0, "loc": 0.0, "scale": 1.0}
    high_snr = np.array([[[20.0]]], dtype=np.float32)
    low_snr = np.array([[[-20.0]]], dtype=np.float32)
    p_high = fit_t_test(params, high_snr)[0, 0, 0]
    p_low = fit_t_test(params, low_snr)[0, 0, 0]
    assert p_high < p_low


def test_truncated_mle_returns_dict_keys():
    params = fit_null_truncated_mle(_clean_samples())
    assert set(params.keys()) == {"df", "loc", "scale"}


def test_truncated_mle_scale_positive():
    params = fit_null_truncated_mle(_clean_samples())
    assert params["scale"] > 0


def test_truncated_mle_df_positive():
    params = fit_null_truncated_mle(_clean_samples())
    assert params["df"] > 0


def test_truncated_mle_empty_input():
    params = fit_null_truncated_mle(np.array([]))
    assert params == {"df": 1.0, "loc": 0.0, "scale": 1.0}

@pytest.mark.parametrize("fitter,label", [
    (fit_null_distribution, "full"),
    (fit_null_truncated_mle, "truncated"),
])
def test_clean_null_parameter_recovery(fitter, label):
    samples = _clean_samples()
    params = fitter(samples)
    assert abs(params["loc"] - TRUE_LOC) < 0.1, f"{label}: loc off"
    assert abs(params["scale"] - TRUE_SCALE) / TRUE_SCALE < 0.10, f"{label}: scale off"
    assert abs(params["df"] - TRUE_DF) / TRUE_DF < 0.20, f"{label}: df off"


def test_contaminated_null_full_fit_is_biased():
    samples = _contaminated_samples()
    params = fit_null_distribution(samples)
    # Scale should be inflated and/or loc shifted relative to true null
    scale_inflated = params["scale"] > TRUE_SCALE * 1.05
    loc_shifted = abs(params["loc"] - TRUE_LOC) > 0.05
    assert scale_inflated or loc_shifted, (
        "Full fit unexpectedly unbiased on contaminated null "
        f"(loc={params['loc']:.3f}, scale={params['scale']:.3f})"
    )


def test_contaminated_null_truncated_recovers():
    samples = _contaminated_samples()
    params = fit_null_truncated_mle(samples)
    assert abs(params["loc"] - TRUE_LOC) < 0.1, f"loc={params['loc']:.3f}"
    assert abs(params["scale"] - TRUE_SCALE) / TRUE_SCALE < 0.10, f"scale={params['scale']:.3f}"
    assert abs(params["df"] - TRUE_DF) / TRUE_DF < 0.20, f"df={params['df']:.3f}"


def test_threshold_stability_truncated():
    clean = _clean_samples()
    contaminated = _contaminated_samples()
    t_clean = compute_significance_threshold(fit_null_truncated_mle(clean))
    t_cont = compute_significance_threshold(fit_null_truncated_mle(contaminated))
    assert abs(t_cont - t_clean) / abs(t_clean) < 0.05, (
        f"Truncated thresholds diverge: clean={t_clean:.3f}, contaminated={t_cont:.3f}"
    )


def test_threshold_instability_full():
    clean = _clean_samples()
    contaminated = _contaminated_samples()
    t_clean = compute_significance_threshold(fit_null_distribution(clean))
    t_cont = compute_significance_threshold(fit_null_distribution(contaminated))
    assert abs(t_cont - t_clean) / abs(t_clean) > 0.05, (
        "Full-fit thresholds unexpectedly stable despite contamination "
        f"(clean={t_clean:.3f}, contaminated={t_cont:.3f})"
    )

def test_pi0_clean_near_one():
    samples = _clean_samples()
    params = fit_null_truncated_mle(samples)
    pi0 = estimate_null_fraction(samples, params["df"], params["loc"], params["scale"])
    assert pi0 > 0.90, f"pi0={pi0:.3f} unexpectedly low on clean null"


def test_pi0_contaminated_below_clean():
    clean = _clean_samples()
    contaminated = _contaminated_samples()
    p_clean = fit_null_truncated_mle(clean)
    p_cont = fit_null_truncated_mle(contaminated)
    pi0_clean = estimate_null_fraction(clean, p_clean["df"], p_clean["loc"], p_clean["scale"])
    pi0_cont = estimate_null_fraction(contaminated, p_cont["df"], p_cont["loc"], p_cont["scale"])
    assert pi0_cont < pi0_clean, (
        f"pi0 not lower on contaminated null: clean={pi0_clean:.3f}, contaminated={pi0_cont:.3f}"
    )


def test_pi0_contaminated_noticeably_below_one():
    samples = _contaminated_samples()
    params = fit_null_truncated_mle(samples)
    pi0 = estimate_null_fraction(samples, params["df"], params["loc"], params["scale"])
    assert pi0 < 0.97, f"pi0={pi0:.3f} not detectably below 1.0 on 10%% contaminated null"
