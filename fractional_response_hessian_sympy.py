"""Symbolic derivation for the fractional-response class-membership Hessian.

This optional helper uses Sympy to derive the scalar-feature, three-class case.
The vector case follows by replacing ``x**2`` with the outer product ``z z'``.
"""

from __future__ import annotations

import sympy as sp  # type: ignore[import-not-found]


def main() -> None:
    x, theta_1, theta_2 = sp.symbols("x theta_1 theta_2")
    w_0, w_1, w_2 = sp.symbols("w_0 w_1 w_2", nonnegative=True)

    eta_1 = x * theta_1
    eta_2 = x * theta_2
    denom = 1 + sp.exp(eta_1) + sp.exp(eta_2)
    p_0 = 1 / denom
    p_1 = sp.exp(eta_1) / denom
    p_2 = sp.exp(eta_2) / denom

    neg_loglik = -(w_0 * sp.log(p_0) + w_1 * sp.log(p_1) + w_2 * sp.log(p_2))
    params = sp.Matrix([theta_1, theta_2])

    gradient = sp.simplify(sp.Matrix([sp.diff(neg_loglik, p) for p in params]))
    hessian = sp.simplify(sp.hessian(neg_loglik, params))

    row_weight = w_0 + w_1 + w_2
    expected_hessian = sp.simplify(
        row_weight
        * x**2
        * sp.Matrix(
            [
                [p_1 * (1 - p_1), -p_1 * p_2],
                [-p_2 * p_1, p_2 * (1 - p_2)],
            ]
        )
    )

    print("Gradient:")
    sp.printing.pprint(gradient)
    print("\nHessian:")
    sp.printing.pprint(hessian)
    print("\nHessian minus expected form:")
    sp.printing.pprint(sp.simplify(hessian - expected_hessian))


if __name__ == "__main__":
    main()
