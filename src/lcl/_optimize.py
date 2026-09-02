from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array, Float64

from lcl.constraints import (
    DEFAULT_NEGATIVE_MIN_ABS,
    pullback_negative_derivatives,
)
from lcl._case_utils import _to_structural_betas
from lcl._struct import OptimizationOptions, OptimizeResult
from lcl.utils import _invert_information


class NewtonState(NamedTuple):
    params: jnp.ndarray
    loss: jnp.ndarray
    grad: jnp.ndarray
    hess: jnp.ndarray
    step_num: int
    error: jnp.ndarray
    failed: jnp.ndarray
    num_fun_eval: jnp.ndarray
    num_grad_hess_eval: jnp.ndarray


def exact_newton_minimize(
    value_fn: Callable[..., Float64[Array, ""]],
    value_grad_hess_fn: Callable[
        ...,
        tuple[
            Float64[Array, ""],
            Float64[Array, "params"],
            Float64[Array, "params params"],
        ],
    ],
    init_params: Float64[Array, "params"],
    *args: object,
    tol: float = 1e-6,
    maxiter: int = 50,
    damping: float = 1e-6,
    max_step_norm: float = 25.0,
    line_search_maxiter: int = 25,
    accept_any_decrease: bool = False,
) -> NewtonState:
    """Minimize a scalar objective with exact Newton steps and Armijo backtracking.

    Parameters
    ----------
    value_fn : Callable[..., Float64[Array, ""]]
        Scalar objective used for line-search evaluations.
    value_grad_hess_fn : Callable
        Function returning a tuple of (loss, gradient, hessian) at current params.
    init_params : Float64[Array, "params"]
        Starting parameter vector.
    *args :
        Additional arguments passed to the objective function (e.g., data, weights).
    tol : float, default=1e-6
        L-infinity norm tolerance for the gradient.
    maxiter : int, default=50
        Maximum number of Newton iterations.
    damping : float, default=1e-6
        Initial relative diagonal shift for the regularized Hessian. The shift is
        increased geometrically if Cholesky factorization does not yield a descent
        direction.
    max_step_norm : float, default=25.0
        Maximum norm allowed for a trial direction before line search.
    line_search_maxiter : int, default=25
        Maximum number of Armijo backtracking iterations per Newton step.
    accept_any_decrease : bool, default=False
        If True, accept a finite step that decreases the objective even when it does
        not satisfy the stricter Armijo sufficient-decrease rule.

    Returns
    -------
    NewtonState
        Final optimizer state containing parameters, value, gradient, Hessian, and
        convergence diagnostics.
    """
    init_loss, init_grad, init_hess = value_grad_hess_fn(init_params, *args)

    init_state = NewtonState(
        params=init_params,
        loss=init_loss,
        grad=init_grad,
        hess=init_hess,
        step_num=0,
        error=jnp.max(jnp.abs(init_grad)),
        failed=jnp.array(False),
        num_fun_eval=jnp.array(0),
        num_grad_hess_eval=jnp.array(1),
    )

    def outer_cond(state: NewtonState) -> jnp.ndarray:
        """Continue while the gradient is too large and iterations remain."""
        return jnp.logical_and(
            jnp.logical_and(state.error > tol, state.step_num < maxiter),
            ~state.failed,
        )

    def outer_body(state: NewtonState) -> NewtonState:
        """Run one damped Newton step plus backtracking line search."""
        # Try a cheap Cholesky factorization first, increasing the diagonal shift
        # geometrically only when the pulled-back Hessian is not positive definite.
        # This avoids an eigenvalue decomposition on the usual well-behaved path.
        H_sym = 0.5 * (state.hess + state.hess.T)
        identity = jnp.eye(state.params.shape[0], dtype=state.params.dtype)
        matrix_scale = jnp.maximum(1.0, jnp.max(jnp.abs(jnp.diag(H_sym))))

        class RegularizationState(NamedTuple):
            shift: jnp.ndarray
            direction: jnp.ndarray
            attempts: int

        def regularized_direction(shift: jnp.ndarray) -> jnp.ndarray:
            """Solve one diagonally shifted Newton system by Cholesky."""
            factor, lower = cho_factor(H_sym + identity * shift)
            return -cho_solve((factor, lower), state.grad)

        initial_shift = jnp.asarray(damping, dtype=state.params.dtype) * matrix_scale
        initial_direction = regularized_direction(initial_shift)
        initial_regularization = RegularizationState(
            initial_shift, initial_direction, 0
        )

        def regularization_cond(reg_state: RegularizationState) -> jnp.ndarray:
            """Increase damping until the direction is finite and descending."""
            valid = jnp.all(jnp.isfinite(reg_state.direction)) & (
                jnp.dot(state.grad, reg_state.direction) < 0.0
            )
            return (~valid) & (reg_state.attempts < 12)

        def regularization_body(reg_state: RegularizationState) -> RegularizationState:
            """Retry the Newton system with ten times more diagonal damping."""
            shift = reg_state.shift * 10.0
            return RegularizationState(
                shift, regularized_direction(shift), reg_state.attempts + 1
            )

        regularization = lax.while_loop(
            regularization_cond, regularization_body, initial_regularization
        )
        newton_direction = regularization.direction
        newton_is_descent = jnp.all(jnp.isfinite(newton_direction)) & (
            jnp.dot(state.grad, newton_direction) < 0.0
        )
        search_direction = jnp.where(newton_is_descent, newton_direction, -state.grad)
        direction_norm = jnp.linalg.norm(search_direction)
        search_direction = search_direction * jnp.minimum(
            1.0, max_step_norm / (direction_norm + 1e-12)
        )
        directional_derivative = jnp.dot(state.grad, search_direction)

        class LSState(NamedTuple):
            step_size: jnp.ndarray
            params: jnp.ndarray
            loss: jnp.ndarray
            ls_iter: int

        def ls_cond(ls_state: LSState) -> jnp.ndarray:
            """Continue backtracking until the candidate is finite and acceptable."""
            expected_improvement = 1e-4 * ls_state.step_size * directional_derivative
            finite_candidate = jnp.isfinite(ls_state.loss) & jnp.all(
                jnp.isfinite(ls_state.params)
            )
            armijo_ok = ls_state.loss <= (state.loss + expected_improvement)
            loss_decreased = ls_state.loss < state.loss
            loss_ok = jnp.where(accept_any_decrease, loss_decreased, armijo_ok)
            return jnp.logical_and(
                ~jnp.logical_and(finite_candidate, loss_ok),
                ls_state.ls_iter < line_search_maxiter,
            )

        def ls_body(ls_state: LSState) -> LSState:
            """Halve the step size and re-evaluate the line-search candidate."""
            new_step = ls_state.step_size * 0.5
            new_params = state.params + new_step * search_direction

            new_loss = value_fn(new_params, *args)

            return LSState(new_step, new_params, new_loss, ls_state.ls_iter + 1)

        # Try the full direction before backtracking.
        full_params = state.params + search_direction
        full_loss = value_fn(full_params, *args)

        init_ls = LSState(
            step_size=jnp.array(1.0),
            params=full_params,
            loss=full_loss,
            ls_iter=0,
        )

        final_ls = lax.while_loop(ls_cond, ls_body, init_ls)

        expected_improvement = 1e-4 * final_ls.step_size * directional_derivative
        finite_candidate = jnp.isfinite(final_ls.loss) & jnp.all(
            jnp.isfinite(final_ls.params)
        )
        armijo_ok = final_ls.loss <= (state.loss + expected_improvement)
        loss_decreased = final_ls.loss < state.loss
        loss_ok = jnp.where(accept_any_decrease, loss_decreased, armijo_ok)
        accepted = jnp.logical_and(
            finite_candidate,
            loss_ok,
        )

        params = jnp.where(accepted, final_ls.params, state.params)

        new_loss, new_grad, new_hess = lax.cond(
            accepted,
            lambda _: value_grad_hess_fn(params, *args),
            lambda _: (state.loss, state.grad, state.hess),
            operand=None,
        )

        return NewtonState(
            params=params,
            loss=new_loss,
            grad=new_grad,
            hess=new_hess,
            step_num=state.step_num + 1,
            error=jnp.max(jnp.abs(new_grad)),
            failed=~accepted,
            num_fun_eval=state.num_fun_eval + final_ls.ls_iter + 1,
            num_grad_hess_eval=state.num_grad_hess_eval + accepted.astype(jnp.int32),
        )

    return lax.while_loop(outer_cond, outer_body, init_state)


def _minimize(
    value_fn: Callable[..., Float64[Array, ""]],
    value_grad_hess_fn: Callable[..., tuple[tuple[Array, Array], Array, Array]],
    params: Float64[Array, "params"],
    args: tuple[object, ...],
    optimization_options: OptimizationOptions | None = None,
    numeraire_idx: int | None = None,
    numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
    assert_converge: bool = False,
    objective_scale: float | Array | None = None,
) -> OptimizeResult:
    """Execute safeguarded exact-Newton maximum-likelihood estimation.

    The objective, analytic gradient, and analytic Hessian are normalized by a
    caller-supplied observational scale. Newton directions use a modified-Cholesky
    diagonal shift, a gradient fallback, a step-norm bound, and Armijo backtracking.

    Parameters
    ----------
    value_fn : Callable
        Scalar negative-loglikelihood used for inexpensive line-search evaluations.
    value_grad_hess_fn : Callable
        Objective returning ``((neg_loglik, score_rows), gradient, hessian)``.
    params : Array
        Initial guess for the unconstrained parameters.
    args : tuple
        Tuple of static and dynamic arguments (e.g., design matrices, weights)
        required by the objective function.
    optimization_options : :class:`~lcl._struct.OptimizationOptions`, optional
        Configuration holding tolerances and maximum iteration limits.
    numeraire_idx : int | None, optional
        Column index of the numeraire variable, if bounded to be strictly negative.
    numeraire_min_abs : float, default=1e-5
        Minimum absolute value imposed on the numeraire coefficient.
    assert_converge : bool, default=False
        If True, raises ``RuntimeError`` if the solver fails to reach the
        specified tolerance.
    objective_scale : float or Array | None, optional
        Positive divisor used to express the stopping gradient per observational
        unit. Defaults to one.

    Returns
    -------
    :class:`~lcl._struct.OptimizeResult`
        Container holding the optimized parameters, the inverse Hessian, case-level
        gradients, and solver diagnostics.
    """
    if optimization_options is None:
        optimization_options = OptimizationOptions()

    # A common per-observation scale gives gradient_tol the same meaning across
    # standalone CL, class-specific M-steps, and demographic M-steps.
    scale_factor = jnp.maximum(
        jnp.asarray(1.0 if objective_scale is None else objective_scale),
        1.0,
    )

    def _value_fn_closure(
        p: Float64[Array, "params"], *inner_args: object
    ) -> Float64[Array, ""]:
        """Evaluate the normalized scalar objective for line search."""
        p_struct = _to_structural_betas(p, numeraire_idx, numeraire_min_abs)
        value = value_fn(p_struct, *inner_args)
        return value / scale_factor

    def _value_grad_hess_closure(
        p: Float64[Array, "params"], *inner_args: object
    ) -> tuple[Array, Array, Array]:
        """Evaluate normalized derivatives in unconstrained parameter space."""
        p_struct = _to_structural_betas(p, numeraire_idx, numeraire_min_abs)
        (val, score_rows), grad_struct, hessian = value_grad_hess_fn(
            p_struct, *inner_args
        )
        grad, _, hessian = pullback_negative_derivatives(
            p, numeraire_idx, grad_struct, score_rows, hessian, numeraire_min_abs
        )
        return val / scale_factor, grad / scale_factor, hessian / scale_factor

    state = exact_newton_minimize(
        _value_fn_closure,
        _value_grad_hess_closure,
        params,
        *args,
        tol=optimization_options.gradient_tol,
        maxiter=optimization_options.maxiter,
        damping=optimization_options.hessian_damping,
        max_step_norm=optimization_options.max_step_norm,
        line_search_maxiter=optimization_options.line_search_maxiter,
        accept_any_decrease=optimization_options.accept_any_decrease,
    )
    params = state.params

    # Translate the low-level stopping state into a public result message.
    error = state.error.item()
    iterations = int(state.step_num)

    if error <= optimization_options.gradient_tol:
        success = True
        message = "Optimization terminated successfully."
    elif bool(state.failed):
        success = False
        message = "Line search failed to find a finite sufficient-decrease step."
    elif iterations >= optimization_options.maxiter:
        success = False
        message = "Maximum number of iterations reached without convergence."
    else:
        success = False
        message = "Optimization halted prematurely."

    if assert_converge and not success:
        raise RuntimeError(message)

    final_eval = value_grad_hess_fn(
        _to_structural_betas(params, numeraire_idx, numeraire_min_abs), *args
    )
    (neg_loglik, grad_n), grad_struct, hessian = final_eval
    grad, grad_n, hessian = pullback_negative_derivatives(
        params, numeraire_idx, grad_struct, grad_n, hessian, numeraire_min_abs
    )
    Hinv, information_diagnostics = _invert_information(
        hessian, label="conditional-logit information matrix"
    )

    return OptimizeResult(
        success=success,
        params=params,
        neg_loglik=neg_loglik,
        message=message,
        hess_inv=Hinv,
        grad_n=grad_n,
        grad=grad,
        nit=iterations,
        nfev=int(state.num_fun_eval + state.num_grad_hess_eval + 1),
        njev=int(state.num_grad_hess_eval + 1),
        information_diagnostics=information_diagnostics,
    )
