import numpy as np
import pytest
from sections.processing import auto_axis, compute_sections, validate_params


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------

def _valid_kwargs(**overrides):
    defaults = dict(
        spacing=0.1,
        prefilter_half_width=2.0,
        edgelock=0.05,
        half_width=0.7,
        right_trim=0.0,
        slope_thr=1.5,
        depth_min=0.0,
        clip_mode="fixed",
    )
    defaults.update(overrides)
    return defaults


def test_validate_params_valid():
    assert validate_params(**_valid_kwargs()) == []


def test_validate_params_auto_clip_mode():
    assert validate_params(**_valid_kwargs(clip_mode="auto")) == []


def test_validate_params_spacing_zero():
    errors = validate_params(**_valid_kwargs(spacing=0))
    assert any("Spacing" in e for e in errors)


def test_validate_params_spacing_negative():
    errors = validate_params(**_valid_kwargs(spacing=-1))
    assert any("Spacing" in e for e in errors)


def test_validate_params_prefilter_zero():
    errors = validate_params(**_valid_kwargs(prefilter_half_width=0))
    assert any("Prefilter" in e for e in errors)


def test_validate_params_edgelock_negative():
    errors = validate_params(**_valid_kwargs(edgelock=-0.1))
    assert any("Edge-lock" in e for e in errors)


def test_validate_params_half_width_zero():
    errors = validate_params(**_valid_kwargs(half_width=0))
    assert any("Half-width" in e for e in errors)


def test_validate_params_right_trim_negative():
    errors = validate_params(**_valid_kwargs(right_trim=-1))
    assert any("Right trim" in e for e in errors)


def test_validate_params_slope_thr_zero():
    errors = validate_params(**_valid_kwargs(slope_thr=0))
    assert any("Slope" in e for e in errors)


def test_validate_params_depth_min_negative():
    errors = validate_params(**_valid_kwargs(depth_min=-1))
    assert any("Depth min" in e for e in errors)


def test_validate_params_invalid_clip_mode():
    errors = validate_params(**_valid_kwargs(clip_mode="bad"))
    assert any("Clip mode" in e for e in errors)


def test_validate_params_multiple_errors():
    errors = validate_params(**_valid_kwargs(spacing=-1, half_width=-1))
    assert len(errors) >= 2


# ---------------------------------------------------------------------------
# auto_axis
# ---------------------------------------------------------------------------

def test_auto_axis_horizontal_trench():
    # Points along x-axis: axis should be roughly horizontal
    x = np.linspace(0, 10, 100)
    y = np.random.default_rng(0).normal(0, 0.01, 100)
    start, end = auto_axis(x, y)
    assert start.shape == (2,)
    assert end.shape == (2,)
    # start should be near x=0, end near x=10
    xs = sorted([start[0], end[0]])
    assert xs[0] < 1.0
    assert xs[1] > 9.0


def test_auto_axis_vertical_trench():
    # Points along y-axis
    x = np.random.default_rng(1).normal(0, 0.01, 100)
    y = np.linspace(0, 10, 100)
    start, end = auto_axis(x, y)
    ys = sorted([start[1], end[1]])
    assert ys[0] < 1.0
    assert ys[1] > 9.0


def test_auto_axis_too_few_points():
    with pytest.raises(ValueError, match="Not enough points"):
        auto_axis(np.array([1.0]), np.array([1.0]))


def test_auto_axis_returns_floats():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    start, end = auto_axis(x, y)
    assert start.dtype == np.float64
    assert end.dtype == np.float64


# ---------------------------------------------------------------------------
# compute_sections
# ---------------------------------------------------------------------------

def _straight_trench(n=200, length=10.0, width=0.5, spacing=1.0):
    """Helper: synthetic straight trench along x-axis."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, length, n)
    y = rng.uniform(-width, width, n)
    z = rng.uniform(-1, 0, n)
    start = np.array([0.0, 0.0])
    end = np.array([length, 0.0])
    return x, y, z, start, end, spacing


def test_compute_sections_returns_dataframe():
    x, y, z, start, end, spacing = _straight_trench()
    df = compute_sections(x, y, z, start, end, spacing)
    assert list(df.columns) == ["section_id", "s", "dist_off", "z"]


def test_compute_sections_section_ids_start_at_zero():
    x, y, z, start, end, spacing = _straight_trench()
    df = compute_sections(x, y, z, start, end, spacing)
    assert df["section_id"].min() == 0


def test_compute_sections_spacing_respected():
    x, y, z, start, end, spacing = _straight_trench(spacing=2.0)
    df = compute_sections(x, y, z, start, end, spacing=2.0)
    # with length=10 and spacing=2, expect ~5 sections
    n_sections = df["section_id"].nunique()
    assert 4 <= n_sections <= 6


def test_compute_sections_prefilter_removes_far_points():
    x, y, z, start, end, spacing = _straight_trench(width=2.0)
    df_narrow = compute_sections(x, y, z, start, end, spacing, prefilter_half_width=0.3)
    df_wide = compute_sections(x, y, z, start, end, spacing, prefilter_half_width=2.0)
    assert len(df_narrow) < len(df_wide)


def test_compute_sections_invalid_spacing():
    x, y, z, start, end, _ = _straight_trench()
    with pytest.raises(ValueError, match="spacing"):
        compute_sections(x, y, z, start, end, spacing=0)


def test_compute_sections_no_nan_in_output():
    x, y, z, start, end, spacing = _straight_trench()
    df = compute_sections(x, y, z, start, end, spacing)
    assert not df.isnull().any().any()
