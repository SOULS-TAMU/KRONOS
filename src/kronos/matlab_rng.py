"""Mersenne Twister generator with MATLAB-compatible ``rand`` and ``randn``.

``rand`` is bit-identical to MATLAB's under the default ``twister`` generator,
so a given seed produces the same multistart starting points. ``randn`` uses a
256-level ziggurat and is used only for the perturbations applied on stagnation
and on multiplier-sign rejection.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = ["MatlabRandom", "ziggurat_table"]

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF


def ziggurat_table() -> np.ndarray:
    """256-level ziggurat layer widths ``X[0..256]`` for the standard normal.

    ``X`` is decreasing: ``X[0]`` is the base strip (which carries the tail),
    ``X[255]`` the narrow cap, ``X[256] = 0``.
    """
    m = 256
    r = 3.6541528853610088
    v = 0.00492867323399
    x = np.zeros(m + 1, dtype=np.float64)
    x[0] = v / math.exp(-0.5 * r * r)
    x[1] = r
    for i in range(2, m):
        prev = x[i - 1]
        x[i] = math.sqrt(-2.0 * math.log(v / prev + math.exp(-0.5 * prev * prev)))
    x[m] = 0.0
    return x


_XTAB = ziggurat_table()


class MatlabRandom:
    """Mersenne Twister stream seeded the way MATLAB's ``rng(seed)`` seeds it.

    Parameters
    ----------
    seed : int
        Equivalent to MATLAB ``rng(seed, 'twister')``.

    Notes
    -----
    ``rand`` is bit-exact against MATLAB.  ``randn`` is bit-exact on the
    ziggurat fast path; see the module docstring.
    """

    def __init__(self, seed: int = 0):
        self._mt = np.zeros(_N, dtype=np.uint32)
        self._mti = _N
        self.seed(seed)

    # -- state -----------------------------------------------------------
    def seed(self, seed: int) -> None:
        mt = self._mt
        mt[0] = np.uint32(seed & 0xFFFFFFFF)
        for i in range(1, _N):
            prev = int(mt[i - 1])
            mt[i] = np.uint32((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        self._mti = _N

    def _twist(self) -> None:
        u64 = np.uint64
        upper, lower, one = u64(_UPPER_MASK), u64(_LOWER_MASK), u64(1)
        mt = self._mt.astype(np.uint64)
        mag01 = np.array([0, _MATRIX_A], dtype=np.uint64)

        y = (mt[0:_N - _M] & upper) | (mt[1:_N - _M + 1] & lower)
        mt[0:_N - _M] = mt[_M:_N] ^ (y >> one) ^ mag01[(y & one).astype(np.intp)]

        y = (mt[_N - _M:_N - 1] & upper) | (mt[_N - _M + 1:_N] & lower)
        mt[_N - _M:_N - 1] = mt[0:_M - 1] ^ (y >> one) ^ mag01[(y & one).astype(np.intp)]

        y = (mt[_N - 1] & upper) | (mt[0] & lower)
        mt[_N - 1] = mt[_M - 1] ^ (y >> one) ^ mag01[int(y & one)]

        self._mt = mt.astype(np.uint32)
        self._mti = 0

    def _words(self, count: int) -> np.ndarray:
        """Next ``count`` tempered 32-bit words."""
        out = np.empty(count, dtype=np.uint64)
        filled = 0
        while filled < count:
            if self._mti >= _N:
                self._twist()
            take = min(count - filled, _N - self._mti)
            out[filled:filled + take] = self._mt[self._mti:self._mti + take]
            self._mti += take
            filled += take

        y = out
        y ^= y >> np.uint64(11)
        y ^= (y << np.uint64(7)) & np.uint64(0x9D2C5680)
        y ^= (y << np.uint64(15)) & np.uint64(0xEFC60000)
        y &= np.uint64(0xFFFFFFFF)
        y ^= y >> np.uint64(18)
        return y

    # -- uniform ---------------------------------------------------------
    def rand(self, *shape: int) -> np.ndarray | float:
        """MATLAB ``rand``: uniform on (0, 1), column-major fill."""
        n = int(np.prod(shape)) if shape else 1
        w = self._words(2 * n)
        a = (w[0::2] >> np.uint64(5)).astype(np.float64)
        b = (w[1::2] >> np.uint64(6)).astype(np.float64)
        vals = (a * 67108864.0 + b) / 9007199254740992.0
        if not shape:
            return float(vals[0])
        return vals.reshape(shape, order="F")

    # -- normal ----------------------------------------------------------
    def randn(self, *shape: int) -> np.ndarray | float:
        """MATLAB ``randn``: standard normal via the 256-level ziggurat."""
        n = int(np.prod(shape)) if shape else 1
        vals = np.empty(n, dtype=np.float64)
        for i in range(n):
            vals[i] = self._randn_scalar()
        if not shape:
            return float(vals[0])
        return vals.reshape(shape, order="F")

    def _randn_scalar(self) -> float:
        x = _XTAB
        for _ in range(200):
            w = self._words(2)
            w1, w2 = int(w[0]), int(w[1])
            u53 = ((w1 >> 5) * 67108864.0 + (w2 >> 6)) / 9007199254740992.0
            u = 2.0 * u53 - 1.0
            j = 255 - (w2 >> 24)

            if abs(u) < x[j + 1] / x[j]:
                return u * x[j]

            if j == 0:
                # Base strip: Marsaglia tail sampling.
                r = x[1]
                while True:
                    e1 = -math.log(max(self.rand(), 5e-324)) / r
                    e2 = -math.log(max(self.rand(), 5e-324))
                    if 2.0 * e2 > e1 * e1:
                        return r + e1 if u > 0 else -(r + e1)
            else:
                z = u * x[j]
                f0 = math.exp(-0.5 * x[j] * x[j])
                f1 = math.exp(-0.5 * x[j + 1] * x[j + 1])
                if f1 + self.rand() * (f0 - f1) < math.exp(-0.5 * z * z):
                    return z
        return 0.0

    # -- convenience -----------------------------------------------------
    def rand_matrix(self, rows: int, cols: int) -> np.ndarray:
        return np.asarray(self.rand(rows, cols)).reshape(rows, cols)
