from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
from equinox import filter_jit
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array, Bool, Float64, Int

from lcl.constraints import NegativeCoefficientBound

from lcl._inference import _invert_information
from lcl.options import OptimizationOptions
from lcl._struct import OptimizeResult


class NewtonDirection(NamedTuple):
    """Direction and local metric, computed once at each accepted iterate."""

    direction: Float64[Array, "params"]
    decrement: Float64[Array, ""]
    diagonal_scale: Float64[Array, "params"]
    active: Bool[Array, "params"]
    shift: Float64[Array, ""]


class NewtonState(NamedTuple):
    """Optimizer iterate with its cached direction and convergence diagnostic."""

    params: Float64[Array, "params"]
    loss: Float64[Array, ""]
    grad: Float64[Array, "params"]
    hess: Float64[Array, "params params"]
    step_num: int
    newton: NewtonDirection
    failed: Bool[Array, ""]
    num_fun_eval: Int[Array, ""]
    num_grad_hess_eval: Int[Array, ""]
    trust_radius: Float64[Array, ""]

    @property
    def error(self) -> Float64[Array, "..."]:
        """Return the cached projected Newton decrement."""
        return self.newton.decrement


def newton_kwargs(
    optimization_options: OptimizationOptions,
) -> dict[str, Any]:
    """Translate an :class:`~lcl.options.OptimizationOptions` into solver kwargs.

    Keeping the translation in one place means every call site -- the standalone
    conditional logit, the class-specific M-step, the membership M-step, and the
    observed-data polish -- reads the same fields, so an option added here reaches
    all of them.

    Parameters
    ----------
    optimization_options : :class:`~lcl.options.OptimizationOptions`
        Solver configuration.

    Returns
    -------
    dict
        Keyword arguments accepted by :func:`exact_newton_minimize`.
    """
    return {
        "tol": optimization_options.newton_decrement_tol,
        "maxiter": optimization_options.maxiter,
        "damping": optimization_options.hessian_damping,
        "max_step_norm": optimization_options.max_step_norm,
        "initial_trust_radius": optimization_options.initial_trust_radius,
        "line_search_maxiter": optimization_options.line_search_maxiter,
        "accept_any_decrease": optimization_options.accept_any_decrease,
    }


def curvature_scaling(
    hess: Float64[Array, "params params"],
) -> tuple[Float64[Array, "params params"], Float64[Array, "params"]]:
    """Symmetrize and standardize local curvature by its absolute diagonal."""
    symmetric = 0.5 * (hess + hess.T)
    diagonal = jnp.abs(jnp.diag(symmetric))
    scale = jnp.sqrt(jnp.where(diagonal > 0.0, diagonal, 1.0))
    return symmetric / (scale[:, None] * scale[None, :]), scale


def projected_active_set(
    params: Float64[Array, "params"],
    grad: Float64[Array, "params"],
    scale: Float64[Array, "params"],
    upper_bounds: Float64[Array, "params"] | None,
) -> Bool[Array, "params"]:
    """Identify near-bound outward slopes in curvature-scaled coordinates.

    The neighborhood shrinks with the projected gradient residual (Bertsekas,
    1982, eqs. 31--34). An inward slope always releases an attained bound.
    """
    if upper_bounds is None:
        return jnp.zeros_like(grad, dtype=bool)
    residual = scale * (params - jnp.minimum(params - grad / scale**2, upper_bounds))
    epsilon = jnp.minimum(0.01, jnp.linalg.norm(residual))
    return ((upper_bounds - params) * scale <= epsilon) & (grad < 0.0)


def _two_metric_hessian(
    scaled_hess: Float64[Array, "params params"],
    active: Bool[Array, "params"],
) -> Float64[Array, "params params"]:
    """Decouple the active coordinates and give them positive diagonal curvature."""
    free_pair = (~active)[:, None] & (~active)[None, :]
    return jnp.where(free_pair, scaled_hess, 0.0) + jnp.diag(
        active.astype(scaled_hess.dtype)
    )


class RegularizationState(NamedTuple):
    """Standardized Cholesky solve and its diagonal shift."""

    shift: Float64[Array, ""]
    direction_scaled: Float64[Array, "params"]
    attempts: int


def regularized_solve(
    scaled_hess: Float64[Array, "params params"],
    scaled_grad: Float64[Array, "params"],
    damping: float,
) -> RegularizationState:
    """Try exact Cholesky first, then diagonal shifts until finite descent.

    An exactly zero slope is stationary even if curvature vanishes, as in a
    zero-mass class. Small nonzero slopes must still solve or report failure.
    """
    identity = jnp.eye(scaled_grad.size, dtype=scaled_grad.dtype)

    def solve(shift: Float64[Array, ""]) -> Float64[Array, "params"]:
        """Solve the shifted standardized system."""
        return -cho_solve(cho_factor(scaled_hess + identity * shift), scaled_grad)

    zero = jnp.asarray(0.0, dtype=scaled_grad.dtype)
    initial = RegularizationState(zero, solve(zero), 0)
    stationary = jnp.all(scaled_grad == 0.0)

    def cond(state: RegularizationState) -> Bool[Array, ""]:
        """Keep regularizing a nonstationary system without a descent direction."""
        valid = jnp.all(jnp.isfinite(state.direction_scaled)) & (
            jnp.dot(scaled_grad, state.direction_scaled) < 0.0
        )
        return (~valid) & (~stationary) & (state.attempts < 12)

    def body(state: RegularizationState) -> RegularizationState:
        """Increase the fallback shift geometrically."""
        fallback = jnp.maximum(
            jnp.asarray(damping, dtype=scaled_grad.dtype),
            jnp.sqrt(jnp.finfo(scaled_grad.dtype).eps),
        )
        shift = jnp.where(state.attempts == 0, fallback, state.shift * 10.0)
        return RegularizationState(shift, solve(shift), state.attempts + 1)

    return lax.while_loop(cond, body, initial)


def projected_decrement(
    params: Float64[Array, "params"],
    feasible_grad: Float64[Array, "params"],
    direction: Float64[Array, "params"],
    active: Bool[Array, "params"],
    upper_bounds: Float64[Array, "params"] | None,
) -> Float64[Array, ""]:
    """Combine free Newton decrease and feasible active-coordinate decrease.

    ``feasible_grad`` omits outward slopes at attained bounds. Only an exactly
    zero feasible slope can bypass a failed solve; invalid descent otherwise
    has infinite decrement, independently of the raw gradient's magnitude.
    """
    valid = jnp.all(jnp.isfinite(direction)) & (jnp.dot(feasible_grad, direction) < 0)
    displacement = direction
    if upper_bounds is not None:
        displacement = jnp.where(
            active, jnp.minimum(params + direction, upper_bounds) - params, direction
        )
    decrease = -jnp.dot(feasible_grad, displacement)
    return jnp.where(
        valid,
        jnp.sqrt(jnp.maximum(decrease, 0.0)),
        jnp.where(jnp.all(feasible_grad == 0.0), 0.0, jnp.inf),
    )


def regularized_newton_direction(
    params: Float64[Array, "params"],
    grad: Float64[Array, "params"],
    hess: Float64[Array, "params params"],
    upper_bounds: Float64[Array, "params"] | None = None,
    damping: float = 0.0,
) -> NewtonDirection:
    """Assemble a scale-equivariant two-metric direction and stopping statistic."""
    scaled_hess, scale = curvature_scaling(hess)
    active = projected_active_set(params, grad, scale, upper_bounds)
    if upper_bounds is not None:
        scaled_hess = _two_metric_hessian(scaled_hess, active)
        # Outward motion at an attained bound cannot consume the trust radius.
        grad = jnp.where(active & (params >= upper_bounds), 0.0, grad)
    solved = regularized_solve(scaled_hess, grad / scale, damping)
    direction = solved.direction_scaled / scale
    decrement = projected_decrement(params, grad, direction, active, upper_bounds)
    return NewtonDirection(direction, decrement, scale, active, solved.shift)


def curvature_step_norm(
    step: Float64[Array, "params"],
    hess: Float64[Array, "params params"],
    newton: NewtonDirection,
) -> Float64[Array, ""]:
    """Measure an accepted displacement in the positive metric used by the solve.

    This is sqrt(step' H step) in the unregularized interior. At active bounds
    it uses the decoupled diagonal block, and when regularized it includes the
    standardized shift. The raw Hessian can be indefinite in a mixture model.
    """
    free_step = jnp.where(newton.active, 0.0, step)
    scaled_step = newton.diagonal_scale * step
    squared = (
        jnp.dot(free_step, hess @ free_step)
        + jnp.sum(jnp.where(newton.active, scaled_step**2, 0.0))
        + newton.shift * jnp.dot(scaled_step, scaled_step)
    )
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def scaled_objective(
    value_fn: Callable[..., Any],
    value_grad_hess_fn: Callable[..., Any],
    scale: int | float | Float64[Array, ""],
    *,
    has_aux: bool = True,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Normalize a value and its derivatives by the same observational mass.

    By default the derivative kernel returns ``((value, scores), grad, hess)``;
    use ``has_aux=False`` for a plain ``(value, grad, hess)`` kernel. Construct
    these closures inside the compiled caller to retain stable JIT cache keys.
    """
    divisor = jnp.maximum(scale, 1.0)

    def value(params: Float64[Array, "params"], *args: object) -> Any:
        """Evaluate the normalized line-search objective."""
        return value_fn(params, *args) / divisor

    def derivatives(params: Float64[Array, "params"], *args: object) -> Any:
        """Evaluate all derivatives on the same objective scale."""
        result, grad, hess = value_grad_hess_fn(params, *args)
        val = result[0] if has_aux else result
        return val / divisor, grad / divisor, hess / divisor

    return value, derivatives


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
    damping: float = 0.0,
    max_step_norm: float = 1000.0,
    initial_trust_radius: float = 1.0,
    line_search_maxiter: int = 40,
    accept_any_decrease: bool = False,
    upper_bounds: Float64[Array, "params"] | None = None,
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
        Tolerance for the Newton decrement. Unlike a raw gradient norm, the
        decrement is invariant to nonsingular diagonal rescaling of parameters.
    maxiter : int, default=50
        Maximum number of Newton iterations.
    damping : float, default=0.0
        Initial diagonal shift in standardized Hessian coordinates. The exact,
        undamped Cholesky solve is always attempted first; this value is used only
        if that solve is not finite and descending.
    max_step_norm : float, default=1000.0
        Maximum adaptive trust radius, measured in the local curvature metric.
        The radius starts at ``initial_trust_radius`` and expands or contracts
        with model agreement.
    initial_trust_radius : float, default=1.0
        Starting trust radius, in the local curvature metric.
    line_search_maxiter : int, default=40
        Maximum number of Armijo backtracking iterations per Newton step.
    accept_any_decrease : bool, default=False
        If True, accept a finite step that decreases the objective even when it does
        not satisfy the stricter Armijo sufficient-decrease rule.
    upper_bounds : Array | None, optional
        Structural upper bounds (infinity for free coordinates). Uses the
        two-metric projected Newton method of Bertsekas (1982), with a diagonal
        metric near binding bounds and the Newton metric on the free block.
        Masks and linear systems retain their full shape under JIT and batching.

    Returns
    -------
    NewtonState
        Final optimizer state containing parameters, value, gradient, Hessian, and
        convergence diagnostics.
    """

    def project(params: Float64[Array, "params"]) -> Float64[Array, "params"]:
        return params if upper_bounds is None else jnp.minimum(params, upper_bounds)

    init_params = project(init_params)
    init_loss, init_grad, init_hess = value_grad_hess_fn(init_params, *args)

    init_newton = regularized_newton_direction(
        init_params, init_grad, init_hess, upper_bounds, damping
    )
    init_state = NewtonState(
        params=init_params,
        loss=init_loss,
        grad=init_grad,
        hess=init_hess,
        step_num=0,
        newton=init_newton,
        failed=jnp.array(False),
        num_fun_eval=jnp.array(0),
        num_grad_hess_eval=jnp.array(1),
        trust_radius=jnp.asarray(
            min(initial_trust_radius, max_step_norm), dtype=init_params.dtype
        ),
    )

    def outer_cond(state: NewtonState) -> Bool[Array, ""]:
        """Continue while the Newton decrement is too large."""
        return jnp.logical_and(
            jnp.logical_and(state.error > tol, state.step_num < maxiter),
            ~state.failed,
        )

    def outer_body(state: NewtonState) -> NewtonState:
        """Run one damped Newton step plus backtracking line search."""
        newton_direction, decrement, diagonal_scale, active, _ = state.newton
        newton_is_descent = jnp.all(jnp.isfinite(newton_direction)) & (
            jnp.dot(state.grad, newton_direction) < 0.0
        )
        # A diagonally preconditioned gradient is the scale-equivariant fallback.
        fallback_direction = -state.grad / (diagonal_scale**2)
        if upper_bounds is not None:
            fallback_direction = (
                project(state.params + fallback_direction) - state.params
            )
        search_direction = jnp.where(
            newton_is_descent, newton_direction, fallback_direction
        )
        direction_norm = jnp.where(
            newton_is_descent,
            decrement,
            jnp.linalg.norm(diagonal_scale * search_direction),
        )
        search_direction = search_direction * jnp.minimum(
            1.0, state.trust_radius / (direction_norm + 1e-12)
        )
        directional_derivative = jnp.dot(state.grad, search_direction)

        def expected_change(
            step_size: Float64[Array, ""], params: Float64[Array, "params"]
        ) -> Float64[Array, ""]:
            if upper_bounds is None:
                return 1e-4 * step_size * directional_derivative
            # Bertsekas (1982), eq. (37): projected active displacement plus
            # the unprojected free displacement. Simply clipping a dense Newton
            # step and using ordinary Armijo can fail even for convex quadratics.
            displacement = jnp.where(
                active, params - state.params, step_size * search_direction
            )
            return 1e-4 * jnp.dot(state.grad, displacement)

        class LSState(NamedTuple):
            step_size: Float64[Array, ""]
            params: Float64[Array, "params"]
            loss: Float64[Array, ""]
            ls_iter: int

        def ls_cond(ls_state: LSState) -> Bool[Array, ""]:
            """Continue backtracking until the candidate is finite and acceptable."""
            expected_improvement = expected_change(ls_state.step_size, ls_state.params)
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
            new_params = project(state.params + new_step * search_direction)

            new_loss = value_fn(new_params, *args)

            return LSState(new_step, new_params, new_loss, ls_state.ls_iter + 1)

        # Try the full direction before backtracking.
        full_params = project(state.params + search_direction)
        full_loss = value_fn(full_params, *args)

        init_ls = LSState(
            step_size=jnp.array(1.0),
            params=full_params,
            loss=full_loss,
            ls_iter=0,
        )

        final_ls = lax.while_loop(ls_cond, ls_body, init_ls)

        expected_improvement = expected_change(final_ls.step_size, final_ls.params)
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

        new_newton = lax.cond(
            accepted,
            lambda _: regularized_newton_direction(
                params, new_grad, new_hess, upper_bounds, damping
            ),
            lambda _: state.newton,
            operand=None,
        )

        # Update the curvature-metric trust radius from agreement between the
        # local quadratic model and the accepted objective change.
        accepted_step = params - state.params
        predicted_decrease = -(
            jnp.dot(state.grad, accepted_step)
            + 0.5 * jnp.dot(accepted_step, state.hess @ accepted_step)
        )
        actual_decrease = state.loss - new_loss
        # A non-positive predicted decrease means the local quadratic model has
        # broken down; that must contract the radius, not expand it.
        agreement = jnp.where(
            predicted_decrease > 0.0,
            actual_decrease
            / jnp.maximum(predicted_decrease, jnp.finfo(state.params.dtype).eps),
            jnp.zeros_like(actual_decrease),
        )
        # The step actually taken is the trust-truncated one, so the
        # "did the step reach the boundary" test must use the truncated length.
        step_metric = jnp.where(
            newton_is_descent,
            curvature_step_norm(accepted_step, state.hess, state.newton),
            jnp.linalg.norm(diagonal_scale * accepted_step),
        )
        contracted_radius = jnp.maximum(0.25 * state.trust_radius, 1e-8)
        expanded_radius = jnp.minimum(2.0 * state.trust_radius, max_step_norm)
        trust_radius = jnp.where(
            (~accepted) | (agreement < 0.25),
            contracted_radius,
            jnp.where(
                (agreement > 0.75) & (step_metric >= 0.9 * state.trust_radius),
                expanded_radius,
                state.trust_radius,
            ),
        )

        return NewtonState(
            params=params,
            loss=new_loss,
            grad=new_grad,
            hess=new_hess,
            step_num=state.step_num + 1,
            newton=new_newton,
            failed=~accepted,
            num_fun_eval=state.num_fun_eval + final_ls.ls_iter + 1,
            num_grad_hess_eval=state.num_grad_hess_eval + accepted.astype(jnp.int32),
            trust_radius=trust_radius,
        )

    return lax.while_loop(outer_cond, outer_body, init_state)


@filter_jit
def _minimize_kernel(
    value_fn: Callable[..., Any],
    value_grad_hess_fn: Callable[..., Any],
    params: Float64[Array, "params"],
    args: tuple[object, ...],
    optimization_options: OptimizationOptions,
    negative_bound: NegativeCoefficientBound,
    scale_factor: Float64[Array, ""],
) -> NewtonState:
    """Compile the complete structural solve once per static configuration.

    Data, starts, weights, and the normalization scale are dynamic leaves. Local
    objective closures are created only during tracing, not as fresh JIT cache
    keys for every standalone fit.
    """
    value, derivatives = scaled_objective(value_fn, value_grad_hess_fn, scale_factor)

    return exact_newton_minimize(
        value,
        derivatives,
        params,
        *args,
        **newton_kwargs(optimization_options),
        upper_bounds=negative_bound.upper_bounds(params),
    )


def _minimize(
    value_fn: Callable[..., Float64[Array, ""]],
    value_grad_hess_fn: Callable[
        ...,
        tuple[
            tuple[Float64[Array, ""], Float64[Array, "cases params"]],
            Float64[Array, "params"],
            Float64[Array, "params params"],
        ],
    ],
    params: Float64[Array, "params"],
    args: tuple[object, ...],
    optimization_options: OptimizationOptions | None = None,
    negative_bound: NegativeCoefficientBound = NegativeCoefficientBound(),
    assert_converge: bool = False,
    objective_scale: float | Float64[Array, ""] | None = None,
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
        Initial coefficient values; the solver projects them onto the bounds.
    args : tuple
        Tuple of static and dynamic arguments (e.g., design matrices, weights)
        required by the objective function.
    optimization_options : :class:`~lcl.options.OptimizationOptions`, optional
        Configuration holding tolerances and maximum iteration limits.
    negative_bound : NegativeCoefficientBound
        Resolved negative coefficient constraint, or an unconstrained record.
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

    # A common per-observation scale gives newton_decrement_tol the same meaning
    # across standalone CL, class-specific M-steps, and demographic M-steps.
    scale_factor = jnp.maximum(
        jnp.asarray(1.0 if objective_scale is None else objective_scale),
        1.0,
    )

    state = _minimize_kernel(
        value_fn,
        value_grad_hess_fn,
        params,
        args,
        optimization_options,
        negative_bound,
        scale_factor,
    )
    params = state.params

    # Translate the low-level stopping state into a public result message.
    error = state.error.item()
    iterations = int(state.step_num)

    if error <= optimization_options.newton_decrement_tol:
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

    final_eval = value_grad_hess_fn(params, *args)
    (neg_loglik, grad_n), grad, hessian = final_eval
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
