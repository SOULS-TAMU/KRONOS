"""Minimum-norm least-squares kernels.

The Newton update solves

    dz = argmin ||dz||_2  subject to  dz in argmin ||J dz - r||_2

Two implementations are available through ``Options.step_method``: ``"pinv"``
(the default, in :mod:`kronos.linalg_svd`) and ``"cod"``, a complete orthogonal
decomposition implemented here as :func:`lsqminnorm`. The remaining methods,
``"lstsq"``, ``"tikhonov"`` and ``"backslash"``, do not compute a minimum-norm
solution and are provided for comparison.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import scipy.linalg as sla

__all__ = [
    "EPS",
    "lsqminnorm",
    "min_norm_solve",
    "null_space",
    "STEP_METHODS",
]

EPS: float = float(np.finfo(np.float64).eps)

STEP_METHODS = ("pinv", "cod", "lstsq", "tikhonov", "backslash")


def _rank_tolerance(shape: tuple[int, int], R: np.ndarray,
                    rule: str = "dense") -> float:
    """Rank tolerance for the complete orthogonal decomposition.

    ``"dense"``   ``max(m, n) * eps * |R[0,0]|``
    ``"sparse"``  ``20 * (m + n) * eps * max_j ||A(:,j)||_2``

    Selected by ``Options.rank_rule``.
    """
    if R.size == 0:
        return 0.0
    scale = abs(float(R[0, 0]))
    m, n = shape
    if rule == "sparse":
        return 20.0 * (m + n) * EPS * scale
    return max(m, n) * EPS * scale


def lsqminnorm(
    A: np.ndarray,
    b: np.ndarray,
    tol: Optional[float] = None,
    rule: str = "dense",
) -> np.ndarray:
    """Minimum-norm least-squares solution of ``A x ~= b``.

    Computed by a complete orthogonal decomposition. Among all ``x``
    minimising ``||A x - b||_2`` this returns the one of smallest ``||x||_2``.

    Parameters
    ----------
    A : (m, n) array
    b : (m,) or (m, k) array
    tol : float, optional
        Rank tolerance applied to the diagonal of the pivoted ``R`` factor.
        Defaults to the ``rule`` below.
    rule : {"dense", "sparse"}
        Which of MATLAB's two ``lsqminnorm`` tolerances to reproduce; see
        :func:`_rank_tolerance`.

    Returns
    -------
    x : (n,) or (n, k) array
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    m, n = A.shape

    vector_rhs = b.ndim == 1
    B = b.reshape(m, -1) if b.size else np.zeros((m, 1))
    if B.shape[0] != m:
        raise ValueError(f"shape mismatch: A is {A.shape}, b is {b.shape}")

    if m == 0 or n == 0:
        X = np.zeros((n, B.shape[1]))
        return X.ravel() if vector_rhs else X

    # Non-finite entries are propagated rather than raising. The iteration
    # relies on this: a NaN step fails the Armijo test, is rejected, the
    # multipliers are damped, and the run continues.
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        X = np.full((n, B.shape[1]), np.nan)
        return X.ravel() if vector_rhs else X

    # First QR with column pivoting:  A[:, piv] = Q @ R
    Q, R, piv = sla.qr(A, mode="economic", pivoting=True)
    k = min(m, n)

    if tol is None:
        tol = _rank_tolerance((m, n), R, rule)

    diag = np.abs(np.diag(R))
    # Leading-block rank count.  |R_ii| is non-increasing under column
    # pivoting, so this equals a plain count, but taking the prefix is robust
    # to the tie-breaking LAPACK occasionally does.
    rank = k
    for i in range(k):
        if diag[i] <= tol:
            rank = i
            break

    X = np.zeros((n, B.shape[1]))
    if rank > 0:
        C = (Q.T @ B)[:rank]
        R1 = R[:rank, :]  # [R11 R12], shape (rank, n)
        if rank == n:
            Y = sla.solve_triangular(R1[:, :rank], C, lower=False)
        else:
            # Second QR annihilates R12:  R1.T = Qz @ Tz  =>  R1 = Tz.T @ Qz.T
            Qz, Tz = sla.qr(R1.T, mode="economic")
            W = sla.solve_triangular(Tz.T, C, lower=True)
            Y = Qz @ W
        X[piv, :] = Y

    return X.ravel() if vector_rhs else X


def min_norm_solve(
    A: np.ndarray,
    b: np.ndarray,
    method: Literal["pinv", "cod", "lstsq", "tikhonov", "backslash"] = "pinv",
    tikhonov_mu: float = 1e-8,
    rcond: Optional[float] = None,
    rule: str = "dense",
    svd_tol_rule: str = "matlab",
) -> np.ndarray:
    """Dispatch the linear step.

    ``"pinv"`` is the default. ``"cod"`` computes the same minimum-norm
    least-squares solution by a different factorisation. The remainder do not
    compute a minimum-norm solution and are provided for comparison.

    - ``"pinv"``      : ``J^dagger r`` from the SVD (default)
    - ``"cod"``       : complete orthogonal decomposition (= MATLAB lsqminnorm)
    - ``"lstsq"``     : LAPACK gelsd least squares
    - ``"tikhonov"``  : ``(A'A + mu I)^-1 A'b``
    - ``"backslash"`` : plain square solve, zeroed if singular (MATLAB ``\\``)
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if method == "pinv":
        from .linalg_svd import minnorm_svd
        return minnorm_svd(A, b, rule=svd_tol_rule)
    if method == "cod":
        return lsqminnorm(A, b, rule=rule)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        return np.full(A.shape[1], np.nan)
    if method == "lstsq":
        return np.linalg.lstsq(A, b, rcond=rcond)[0]
    if method == "tikhonov":
        AtA = A.T @ A
        n = AtA.shape[0]
        try:
            return np.linalg.solve(AtA + tikhonov_mu * np.eye(n), A.T @ b)
        except np.linalg.LinAlgError:
            return np.zeros(A.shape[1])
    if method == "backslash":
        try:
            if A.shape[0] == A.shape[1]:
                with np.errstate(all="ignore"):
                    x = np.linalg.solve(A, b)
            else:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            if not np.all(np.isfinite(x)):
                return np.zeros(A.shape[1])
            return x
        except np.linalg.LinAlgError:
            return np.zeros(A.shape[1])

    raise ValueError(f"unknown step method {method!r}; expected one of {STEP_METHODS}")


def null_space(A: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """Orthonormal basis for the null space of ``A``.

    Computed from the SVD with ``tol = max(m, n) * eps(sigma_max)``. Used by the
    second-order classification, where the reduced Hessian is formed on the null
    space of the active constraint Jacobian.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    if A.size == 0:
        return np.eye(A.shape[1]) if A.shape[1] else np.zeros((0, 0))

    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    if tol is None:
        smax = float(s[0]) if s.size else 0.0
        tol = max(A.shape) * np.spacing(smax) if smax > 0 else 0.0

    rank = int(np.sum(s > tol))
    return Vt[rank:].T.copy()
