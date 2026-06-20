from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from equinox import combine, is_array, partition
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve
from jaxopt import BFGS  # type: ignore[import-untyped]
from jaxtyping import Array, Float64

from lcl.constraints import (
    DEFAULT_NEGATIVE_MIN_ABS,
    pullback_negative_derivatives,
    pullback_negative_gradient,
    pullback_negative_score_rows,
)
from lcl._case_utils import _to_structural_betas
from lcl._struct import MleConfig, OptimizeResult


class NewtonState(NamedTuple):
    params: jnp.ndarray
    loss: jnp.ndarray
    grad: jnp.ndarray
    hess: jnp.ndarray
    step_num: int
    error: jnp.ndarray


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
        Tikhonov regularization (ridge penalty) applied to the diagonal of the Hessian.
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
    )

    def outer_cond(state: NewtonState) -> jnp.ndarray:
        """Continue while the gradient is too large and iterations remain."""
        return jnp.logical_and(state.error > tol, state.step_num < maxiter)

    def outer_body(state: NewtonState) -> NewtonState:
        """Run one damped Newton step plus backtracking line search."""
        # Use Cholesky rather than LU because the damped Hessian is positive
        # definite; this is materially faster for the small systems optimized here.
        H_sym = 0.5 * (state.hess + state.hess.T)
        H_damped = H_sym + jnp.eye(state.params.shape[0]) * damping

        # JAX Cholesky solve
        c, lower = cho_factor(H_damped)
        newton_direction = -cho_solve((c, lower), state.grad)

        newton_is_descent = jnp.logical_and(
            jnp.all(jnp.isfinite(newton_direction)),
            jnp.dot(state.grad, newton_direction) < 0.0,
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

        # Start line search
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
        )

    return lax.while_loop(outer_cond, outer_body, init_state)


def _minimize(
    loglik_fn: Callable[
        ...,
        tuple[tuple[Array, Array], Array] | tuple[tuple[Array, Array], Array, Array],
    ],
    params: Float64[Array, "params"],
    args: tuple[object, ...],
    mle_config: MleConfig | None = None,
    numeraire_idx: int | None = None,
    numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS,
    assert_converge: bool = False,
) -> OptimizeResult:
    """Execute the L-BFGS optimization routine for Maximum Likelihood Estimation.

    Employs JAXopt's hardware-accelerated L-BFGS solver with zoom line-search.
    Crucially, this function performs an internal scaling normalization to prevent
    the optimizer from taking disastrously large initial steps that could push
    latent variables into the zero-gradient region of the softplus transformation.

    Parameters
    ----------
    loglik_fn : Callable
        The objective function returning a tuple of `((neg_loglik, aux), gradient)`.
    params : Array
        Initial guess for the unconstrained parameters.
    args : tuple
        Tuple of static and dynamic arguments (e.g., design matrices, weights)
        required by the objective function.
    mle_config : :class:`~lcl._struct.MleConfig`, optional
        Configuration holding tolerances and maximum iteration limits.
    numeraire_idx : int | None, optional
        Column index of the numeraire variable, if bounded to be strictly negative.
    numeraire_min_abs : float, default=1e-5
        Minimum absolute value imposed on the numeraire coefficient.
    assert_converge : bool, default=False
        If True, throws an AssertionError if the solver fails to reach the
        specified tolerance.

    Returns
    -------
    :class:`~lcl._struct.OptimizeResult`
        Container holding the optimized parameters, the inverse Hessian, case-level
        gradients, and solver diagnostics.
    """

    if mle_config is None:
        mle_config = MleConfig()

    dynamic_args, static_args = partition(args, is_array)

    # Evaluate the objective once to get a scaling factor.
    # This prevents BFGS from taking disastrously large initial steps
    # that push latent variables into the softplus vanishing gradient zone.
    p_struct_init = _to_structural_betas(params, numeraire_idx, numeraire_min_abs)
    init_eval = loglik_fn(p_struct_init, *args)
    init_res = init_eval[0]
    init_val = init_res[0] if isinstance(init_res, tuple) else init_res
    scale_factor = jnp.maximum(jnp.abs(init_val), 1.0)

    def _loglik_fn_closure(
        p: Float64[Array, "params"], *dyn_args: object
    ) -> tuple[tuple[Array, Array], Array]:
        """Scale the objective and apply the numeraire chain rule for JAXopt."""
        p_struct = _to_structural_betas(p, numeraire_idx, numeraire_min_abs)
        all_args = combine(dyn_args, static_args)

        # Obtain the analytical gradient
        loglik_eval = loglik_fn(p_struct, *all_args)
        (val, aux), grad = loglik_eval[:2]

        grad = pullback_negative_gradient(p, numeraire_idx, grad)
        aux = pullback_negative_score_rows(p, numeraire_idx, aux)

        # Normalize objective and gradient internally
        return (val / scale_factor, aux), grad / scale_factor

    solver = BFGS(
        fun=_loglik_fn_closure,
        value_and_grad=True,
        has_aux=True,
        linesearch="zoom",
        max_stepsize=1.0,
        maxiter=mle_config.maxiter,
        tol=mle_config.ftol,
        verbose=False,
        implicit_diff=False,
    )

    params, state = solver.run(params, *dynamic_args)

    # Check for convergence
    error = state.error.item()
    stepsize = state.stepsize.item()
    iterations = state.iter_num.item()

    if error <= mle_config.ftol:
        success = True
        message = "Optimization terminated successfully."
    elif iterations >= mle_config.maxiter:
        success = False
        message = "Maximum number of iterations reached without convergence."
    elif stepsize <= 1e-8:
        success = False
        message = "Line search failed."
    else:
        success = False
        message = "Optimization halted prematurely."

    if assert_converge and not success:
        raise RuntimeError(message)

    grad_n = state.aux

    final_eval = loglik_fn(
        _to_structural_betas(params, numeraire_idx, numeraire_min_abs), *args
    )
    if len(final_eval) == 3:
        (_, grad_n_unscaled), grad_struct, hessian = final_eval
        grad_n = grad_n_unscaled
        _, grad_n, hessian = pullback_negative_derivatives(
            params, numeraire_idx, grad_struct, grad_n, hessian
        )
        Hinv = jnp.linalg.pinv(0.5 * (hessian + hessian.T))
    else:
        Hinv = jnp.linalg.pinv(jnp.dot(grad_n.T, grad_n))

    return OptimizeResult(
        success=success,
        params=params,
        neg_loglik=state.value * scale_factor,  # Unscale back to real magnitude
        message=message,
        hess_inv=Hinv,
        grad_n=grad_n,
        grad=state.grad * scale_factor,  # Unscale back to real magnitude
        nit=iterations,
        nfev=state.num_fun_eval.item(),
        njev=state.num_grad_eval.item(),
    )
