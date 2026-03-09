import numpy as np
from scipy.stats import t as scipy_t

from quantify.statistical_model import (
    compute_significance_threshold,
    fit_null_distribution,
    fit_t_test,
)

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
    null_samples = np.random.normal(size=200)
    snr_map = np.random.normal(size=(4, 5, 6)).astype(np.float32)
    p_values = fit_t_test(null_samples, snr_map)
    assert p_values.shape == snr_map.shape


def test_fit_t_test_values_in_unit_interval():
    null_samples = np.random.normal(size=300)
    snr_map = np.random.normal(size=(3, 3, 3)).astype(np.float32)
    p_values = fit_t_test(null_samples, snr_map)
    assert np.all(p_values >= 0.0)
    assert np.all(p_values <= 1.0)


def test_fit_t_test_high_snr_gives_low_pvalue():
    null_samples = np.random.normal(loc=0.0, scale=1.0, size=500)
    high_snr = np.array([[[20.0]]], dtype=np.float32)
    low_snr = np.array([[[-20.0]]], dtype=np.float32)
    p_high = fit_t_test(null_samples, high_snr)[0, 0, 0]
    p_low = fit_t_test(null_samples, low_snr)[0, 0, 0]
    assert p_high < p_low


def test_fit_t_test_empty_null_returns_zeros():
    snr_map = np.ones((2, 2, 2), dtype=np.float32)
    result = fit_t_test(np.array([]), snr_map)
    assert np.all(result == 0.0)
