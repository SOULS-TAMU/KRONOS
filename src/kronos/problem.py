"""Problem definition and squared-slack reformulation.

KRONOS solves equality-constrained problems

    min f(P)   s.t.  H(P) = 0

so inequalities and variable bounds are folded into ``H`` with squared slacks
before the solver ever sees them:

    g_i(x) <= 0   ->   g_i(x) + s_i**2 = 0        sigma_i = +1
    g_i(x) >= 0   ->   g_i(x) - s_i**2 = 0        sigma_i = -1
    x_j >= l_j    ->   x_j - l_j - s**2  = 0      sigma   = -1
    x_j <= u_j    ->   u_j - x_j - s**2  = 0      sigma   = -1

The sign ``sigma_i`` and the row/slack correspondence are recorded so the
multiplier of the original inequality can be recovered as ``mu_i = sigma_i *
nu_i`` and checked for dual feasibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import sympy as sp

__all__ = ["Problem", "SlackRow", "parse_expr"]


def parse_expr(expr: Any, symbols: Sequence[sp.Symbol]) -> sp.Expr:
    """Coerce a string / number / sympy object into a sympy expression.

    Strings use MATLAB-ish syntax: ``^`` is accepted as a power operator.
    """
    if isinstance(expr, sp.Expr):
        return expr
    if isinstance(expr, (int, float, np.integer, np.floating)):
        return sp.Float(float(expr))
    if isinstance(expr, str):
        local = {s.name: s for s in symbols}
        return sp.sympify(expr.replace("^", "**"), locals=local)
    raise TypeError(f"cannot interpret {type(expr).__name__} as an expression")


@dataclass(frozen=True)
class SlackRow:
    """A row of ``H`` that came from an inequality, and its squared slack."""

    row: int          # index into Problem.h
    sign: int         # +1 for  g <= 0  (row is g + s**2), -1 for  g >= 0
    slack_var: int    # index into Problem.var_names of the slack variable
    origin: str = ""  # human-readable provenance, e.g. "g[2]" or "x[4] >= lb"


@dataclass
class Problem:
    """An NLP in KRONOS's internal equality-only form.

    Attributes
    ----------
    name : str
    var_names : list of str
        Decision variables *including* any squared-slack variables.
    f : sympy expression
    h : list of sympy expressions
        Equality constraints; ``h_i(P) = 0``.
    lb, ub : ndarray
        Wide numerical safety clamps, not the modelled bounds.  Real bounds
        live in ``h`` as squared-slack rows.
    x0 : ndarray, shape (n,) or (n, K)
        Starting point. A 2-D array supplies K explicit multistart columns and
        suppresses the random scatter.
    slack_rows : list of SlackRow
    fstar : float, optional
        Known global optimum, enabling the global-hit metric.
    """

    name: str
    var_names: list[str]
    f: sp.Expr
    h: list[sp.Expr] = field(default_factory=list)
    lb: np.ndarray = None       # type: ignore[assignment]
    ub: np.ndarray = None       # type: ignore[assignment]
    x0: np.ndarray = None       # type: ignore[assignment]
    slack_rows: list[SlackRow] = field(default_factory=list)
    fstar: Optional[float] = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        n = len(self.var_names)
        self.lb = np.full(n, -1e10) if self.lb is None else np.asarray(self.lb, float).ravel()
        self.ub = np.full(n, +1e10) if self.ub is None else np.asarray(self.ub, float).ravel()
        if self.x0 is None:
            self.x0 = np.zeros(n)
        else:
            x0 = np.asarray(self.x0, float)
            if x0.ndim == 1 and x0.size != n and x0.size % n == 0:
                x0 = x0.reshape(n, -1, order="F")   # flattened start matrix
            self.x0 = x0
        for attr in ("lb", "ub"):
            if getattr(self, attr).size != n:
                raise ValueError(f"{attr} has size {getattr(self, attr).size}, expected {n}")
        if self.x0.shape[0] != n:
            raise ValueError(f"x0 has {self.x0.shape[0]} rows, expected {n}")

    @property
    def n(self) -> int:
        """Number of decision variables, slacks included."""
        return len(self.var_names)

    @property
    def m(self) -> int:
        """Number of equality constraints."""
        return len(self.h)

    @property
    def symbols(self) -> list[sp.Symbol]:
        return [sp.Symbol(v, real=True) for v in self.var_names]

    @property
    def ineq_row_mask(self) -> np.ndarray:
        mask = np.zeros(self.m, dtype=bool)
        for sr in self.slack_rows:
            mask[sr.row] = True
        return mask

    @property
    def ineq_row_sign(self) -> np.ndarray:
        sign = np.zeros(self.m)
        for sr in self.slack_rows:
            sign[sr.row] = sr.sign
        return sign

    @property
    def slack_var_index(self) -> np.ndarray:
        idx = np.full(self.m, -1, dtype=int)
        for sr in self.slack_rows:
            idx[sr.row] = sr.slack_var
        return idx

    def __repr__(self) -> str:
        return (f"Problem({self.name!r}, n={self.n}, m={self.m}, "
                f"slack_rows={len(self.slack_rows)})")

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        name: str,
        variables: Sequence[str] | int,
        objective: Any,
        equalities: Iterable[Any] = (),
        inequalities: Iterable[Any] = (),
        inequality_sense: str | Sequence[str] = "<=",
        lb: Optional[Sequence[float]] = None,
        ub: Optional[Sequence[float]] = None,
        x0: Optional[Sequence[float]] = None,
        fstar: Optional[float] = None,
        bounds_as_slacks: bool = True,
    ) -> "Problem":
        """Build a problem from its natural statement, adding squared slacks.

        Parameters
        ----------
        variables : list of names, or an int ``n`` for ``x1..xn``
        objective : str or sympy expression
        equalities : iterable of expressions, each meaning ``expr == 0``
        inequalities : iterable of expressions, each meaning ``expr <= 0``
            (or ``>= 0``, per ``inequality_sense``)
        inequality_sense : ``"<="`` / ``">="``, or one per inequality
        lb, ub : modelled variable bounds
        bounds_as_slacks : bool
            Convert finite ``lb``/``ub`` entries into squared-slack rows.
            When False the bounds are kept as numerical clamps only.

        Examples
        --------
        >>> p = Problem.build(
        ...     "hs71", ["x1", "x2", "x3", "x4"],
        ...     objective="x1*x4*(x1 + x2 + x3) + x3",
        ...     equalities=["x1**2 + x2**2 + x3**2 + x4**2 - 40"],
        ...     inequalities=["25 - x1*x2*x3*x4"],
        ...     lb=[1, 1, 1, 1], ub=[5, 5, 5, 5],
        ...     x0=[1, 5, 5, 1])
        >>> p.n, p.m, len(p.slack_rows)
        (13, 10, 9)
        """
        if isinstance(variables, int):
            names = [f"x{i + 1}" for i in range(variables)]
        else:
            names = list(variables)
        n_user = len(names)

        syms = [sp.Symbol(v, real=True) for v in names]
        f_expr = parse_expr(objective, syms)
        h_exprs = [parse_expr(e, syms) for e in equalities]

        ineq = [parse_expr(e, syms) for e in inequalities]
        senses = ([inequality_sense] * len(ineq) if isinstance(inequality_sense, str)
                  else list(inequality_sense))
        if len(senses) != len(ineq):
            raise ValueError("inequality_sense must be a single string or one per inequality")

        lb_arr = np.full(n_user, -np.inf) if lb is None else np.asarray(lb, float).ravel()
        ub_arr = np.full(n_user, +np.inf) if ub is None else np.asarray(ub, float).ravel()
        x0_arr = np.zeros(n_user) if x0 is None else np.asarray(x0, float).ravel()

        slack_rows: list[SlackRow] = []
        extra_names: list[str] = []
        extra_x0: list[float] = []

        def new_slack(tag: str) -> sp.Symbol:
            nm = f"s{len(extra_names) + 1}"
            extra_names.append(nm)
            extra_x0.append(1.0)
            return sp.Symbol(nm, real=True)

        # inequalities -> squared slacks
        for k, (g, sense) in enumerate(zip(ineq, senses)):
            s = new_slack(f"g[{k}]")
            if sense == "<=":
                h_exprs.append(g + s ** 2)
                sign = +1
            elif sense == ">=":
                h_exprs.append(g - s ** 2)
                sign = -1
            else:
                raise ValueError(f"inequality_sense must be '<=' or '>=', got {sense!r}")
            slack_rows.append(SlackRow(len(h_exprs) - 1, sign,
                                       n_user + len(extra_names) - 1, f"g[{k}] {sense} 0"))

        # variable bounds -> squared slacks
        if bounds_as_slacks:
            for j in range(n_user):
                if np.isfinite(lb_arr[j]):
                    s = new_slack(f"x[{j}]>=lb")
                    h_exprs.append(syms[j] - sp.Float(float(lb_arr[j])) - s ** 2)
                    slack_rows.append(SlackRow(len(h_exprs) - 1, -1,
                                               n_user + len(extra_names) - 1,
                                               f"{names[j]} >= {lb_arr[j]:g}"))
                if np.isfinite(ub_arr[j]):
                    s = new_slack(f"x[{j}]<=ub")
                    h_exprs.append(sp.Float(float(ub_arr[j])) - syms[j] - s ** 2)
                    slack_rows.append(SlackRow(len(h_exprs) - 1, -1,
                                               n_user + len(extra_names) - 1,
                                               f"{names[j]} <= {ub_arr[j]:g}"))
            clamp_lo = np.full(n_user + len(extra_names), -1e10)
            clamp_hi = np.full(n_user + len(extra_names), +1e10)
        else:
            clamp_lo = np.concatenate([np.where(np.isfinite(lb_arr), lb_arr, -1e10),
                                       np.full(len(extra_names), -1e10)])
            clamp_hi = np.concatenate([np.where(np.isfinite(ub_arr), ub_arr, 1e10),
                                       np.full(len(extra_names), 1e10)])

        # initialise slacks consistently with x0 where we can
        all_names = names + extra_names
        full_x0 = np.concatenate([x0_arr, np.asarray(extra_x0, float)])
        subs = dict(zip(syms, x0_arr))
        for sr in slack_rows:
            row = h_exprs[sr.row]
            s_sym = sp.Symbol(all_names[sr.slack_var], real=True)
            base = row.subs({s_sym: 0}).subs(subs)
            try:
                val = float(base)
            except (TypeError, ValueError):
                continue
            # row is base +/- s**2 = 0  ->  s = sqrt(-/+ base) when feasible
            target = -val if sr.sign > 0 else val
            full_x0[sr.slack_var] = float(np.sqrt(target)) if target > 0 else 0.0

        return cls(
            name=name,
            var_names=all_names,
            f=f_expr,
            h=h_exprs,
            lb=clamp_lo,
            ub=clamp_hi,
            x0=full_x0,
            slack_rows=slack_rows,
            fstar=fstar,
            meta={"n_user_vars": n_user, "n_user_ineq": len(ineq)},
        )

    # ------------------------------------------------------------------
    def detect_slack_rows(self) -> list[SlackRow]:
        """Recover the squared-slack structure of ``h`` symbolically.

        Used for problems supplied in already-reformulated form, where the
        correspondence between rows and slack variables is implicit. A variable
        ``s`` is a squared slack for row ``i`` when it does not appear in ``f``,
        appears in no row other than ``i``, and ``dh_i/ds = +/-2 s``.
        """
        syms = self.symbols
        f_free = self.f.free_symbols
        rows_free = [e.free_symbols for e in self.h]
        found: list[SlackRow] = []
        for j, s in enumerate(syms):
            if s in f_free:
                continue
            hits = [i for i, fr in enumerate(rows_free) if s in fr]
            if len(hits) != 1:
                continue
            i = hits[0]
            d = sp.simplify(sp.diff(self.h[i], s))
            if d == 2 * s:
                found.append(SlackRow(i, +1, j, f"detected h[{i}]"))
            elif d == -2 * s:
                found.append(SlackRow(i, -1, j, f"detected h[{i}]"))
        # one slack per row
        seen: dict[int, SlackRow] = {}
        for sr in found:
            seen.setdefault(sr.row, sr)
        return sorted(seen.values(), key=lambda r: r.row)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "var_names": self.var_names,
            "f": str(self.f),
            "h": [str(e) for e in self.h],
            "lb": self.lb.tolist(),
            "ub": self.ub.tolist(),
            "x0": np.asarray(self.x0).tolist(),
            "slack_rows": [[s.row, s.sign, s.slack_var, s.origin] for s in self.slack_rows],
            "fstar": self.fstar,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Problem":
        names = list(d["var_names"])
        syms = [sp.Symbol(v, real=True) for v in names]
        return cls(
            name=d["name"],
            var_names=names,
            f=parse_expr(d["f"], syms),
            h=[parse_expr(e, syms) for e in d["h"]],
            lb=np.asarray(d["lb"], float),
            ub=np.asarray(d["ub"], float),
            x0=np.asarray(d["x0"], float),
            slack_rows=[SlackRow(*row) for row in d.get("slack_rows", [])],
            fstar=d.get("fstar"),
            meta=d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "Problem":
        return cls.from_dict(json.loads(Path(path).read_text()))
