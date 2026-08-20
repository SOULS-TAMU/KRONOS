"""MATLAB-compatible random numbers -- reproducing the reference multistart."""
import numpy as np

from kronos.core import scatter_starts
from kronos.matlab_rng import MatlabRandom


# MATLAB R2024b:  rng(42, 'twister'); rand(1, 6)
MATLAB_RAND_42 = [0.37454011884736249, 0.95071430640991617, 0.73199394181140509,
                  0.5986584841970366, 0.15601864044243652, 0.15599452033620265]


def test_rand_is_bit_identical_to_matlab():
    got = np.asarray(MatlabRandom(42).rand(6))
    assert np.array_equal(got, np.array(MATLAB_RAND_42))


def test_rand_matrix_fills_column_major_like_matlab():
    m = np.asarray(MatlabRandom(42).rand(3, 2))
    assert np.array_equal(m[:, 0], np.array(MATLAB_RAND_42[:3]))
    assert np.array_equal(m[:, 1], np.array(MATLAB_RAND_42[3:6]))


def test_streams_are_reproducible():
    assert np.array_equal(np.asarray(MatlabRandom(7).rand(50)),
                          np.asarray(MatlabRandom(7).rand(50)))


def test_randn_is_standard_normal():
    x = np.asarray(MatlabRandom(7).randn(20000))
    assert abs(x.mean()) < 0.05
    assert abs(x.std(ddof=1) - 1.0) < 0.05
    assert x.min() < -3.0 and x.max() > 3.0


def test_scatter_keeps_x0_as_first_start_and_respects_real_bounds():
    lb = np.array([-1.0, -1e10])
    ub = np.array([2.0, 1e10])
    x0 = np.array([0.5, 0.25])
    S = scatter_starts(x0, lb, ub, 8, ms_scale=1.5, rng=MatlabRandom(42))
    assert S.shape == (2, 8)
    assert np.array_equal(S[:, 0], x0)
    assert np.all(S[0, 1:] >= lb[0]) and np.all(S[0, 1:] <= ub[0])
    # unbounded coordinate is scattered in x0 +/- 5 * ms_scale
    assert np.all(np.abs(S[1, 1:] - x0[1]) <= 5 * 1.5 + 1e-9)
