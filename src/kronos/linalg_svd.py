"""Minimum-norm least-squares by the Moore-Penrose pseudoinverse.

Computes the Newton and feasibility-restoration steps as

    dz    = -J^dagger r
    dP    = -alpha_feas * J_H^dagger H

with ``^dagger`` the Moore-Penrose pseudoinverse obtained from the SVD. This is
the default step, selected by ``step_method="pinv"``. The complete orthogonal
decomposition in :mod:`kronos.linalg` computes the same quantity by a different
factorisation.

Rank tolerance
--------------
The pseudoinverse is not defined without a numerical rank, and the choice of
cutoff has a large effect on ill-conditioned systems. Three conventions are
available through ``Options.svd_tol_rule``:

``"matlab"``   ``tol = max(m, n) * eps * sigma_max``, the default, matching
               MATLAB's ``pinv``.
``"numpy"``    ``tol = 1e-15 * sigma_max``, matching NumPy's ``pinv``.
``"exact"``    ``tol = 0``, inverting every nonzero singular value. This is the
               literal definition and is numerically unusable on a
               rank-deficient system; it is included for completeness.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

__all__ = ["EPS", "pinv", "minnorm_svd", "svd_rank_tolerance", "SVD_TOL_RULES"]

EPS: float = float(np.finfo(np.float64).eps)

SVD_TOL_RULES = ("matlab", "numpy", "exact")


def svd_rank_tolerance(shape: tuple[int, int], s: np.ndarray,
                       rule: str = "matlab") -> float:
    """Singular-value cutoff below which a direction is treated as null."""
    if s.size == 0:
        return 0.0
    smax = float(s[0])
    if rule == "exact":
        return 0.0
    if rule == "numpy":
        return 1e-15 * smax
    return max(shape) * EPS * smax          # "matlab": MATLAB pinv default


def pinv(A: np.ndarray, tol: Optional[float] = None,
         rule: str = "matlab") -> np.ndarray:
    """Moore-Penrose pseudoinverse of ``A`` from its SVD.

    ``A = U S V^T``  =>  ``A^dagger = V S^dagger U^T`` where ``S^dagger``
    inverts each singular value above ``tol`` and zeroes the rest.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    m, n = A.shape
    if m == 0 or n == 0:
        return np.zeros((n, m))
    if not np.all(np.isfinite(A)):
        return np.full((n, m), np.nan)

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if tol is None:
        tol = svd_rank_tolerance((m, n), s, rule)

    s_inv = np.zeros_like(s)
    keep = s > tol
    s_inv[keep] = 1.0 / s[keep]
    return (Vt.T * s_inv) @ U.T


def minnorm_svd(A: np.ndarray, b: np.ndarray, tol: Optional[float] = None,
                rule: Literal["matlab", "numpy", "exact"] = "matlab",
                return_rank: bool = False):
    """``A^dagger b``, the minimum-norm least-squares solution via the SVD.

    Equivalent to ``pinv(A) @ b`` but formed without materialising the
    pseudoinverse, which is faster and slightly more accurate.

    Parameters
    ----------
    A : (m, n) array
    b : (m,) or (m, k) array
    tol : float, optional
        Explicit singular-value cutoff; overrides ``rule``.
    rule : {"matlab", "numpy", "exact"}
        How to pick the cutoff when ``tol`` is None.  See the module docstring.
    return_rank : bool
        Also return the numerical rank that was used.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape
    vector_rhs = b.ndim == 1
    B = b.reshape(m, -1) if b.size else np.zeros((m, 1))

    if m == 0 or n == 0:
        X = np.zeros((n, B.shape[1]))
        out = X.ravel() if vector_rhs else X
        return (out, 0) if return_rank else out

    # MATLAB propagates non-finite input rather than raising; the KKT
    # iteration relies on a NaN step being rejected by the line search.
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        X = np.full((n, B.shape[1]), np.nan)
        out = X.ravel() if vector_rhs else X
        return (out, 0) if return_rank else out

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if tol is None:
        tol = svd_rank_tolerance((m, n), s, rule)

    keep = s > tol
    rank = int(np.count_nonzero(keep))
    X = np.zeros((n, B.shape[1]))
    if rank:
        # x = V_r diag(1/s_r) U_r^T b
        c = U[:, :rank].T @ B
        X = Vt[:rank].T @ (c / s[:rank, None])

    out = X.ravel() if vector_rhs else X
    return (out, rank) if return_rank else out
