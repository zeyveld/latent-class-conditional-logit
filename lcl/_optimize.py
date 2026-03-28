from typing import Callable

import jax.numpy as jnp
from equinox import combine, is_array, partition
from jax import Array
from jax.nn import sigmoid
from jaxopt import BFGS

from lcl._case_utils import _to_structural_betas
from lcl._struct import MleConfig, OptimizeResult


def _minimize(
    loglik_fn: Callable,
    params: Array,
    args: tuple,
    mle_config: MleConfig = MleConfig(),
    numeraire_idx: int | None = None,
    assert_converge=False,
) -> OptimizeResult:
    """LBFGS optimization routine."""

    dynamic_args, static_args = partition(args, is_array)

    # --- NORMALIZATION FIX ---
    # Evaluate the objective once to get a scaling factor.
    # This prevents BFGS from taking disastrously large initial steps
    # that push latent variables into the softplus vanishing gradient zone.
    p_struct_init = _to_structural_betas(params, numeraire_idx)
    init_res, _ = loglik_fn(p_struct_init, *args)
    init_val = init_res[0] if isinstance(init_res, tuple) else init_res
    scale_factor = jnp.clip(jnp.abs(init_val), a_min=1.0)

    def _loglik_fn_closure(p, *dyn_args) -> tuple[tuple[Array, Array], Array]:
        p_struct = _to_structural_betas(p, numeraire_idx)
        all_args = combine(dyn_args, static_args)

        # Obtain the analytical gradient
        (val, aux), grad = loglik_fn(p_struct, *all_args)

        # Apply chain rule for numeraire
        if numeraire_idx is not None:
            derivative = sigmoid(p[numeraire_idx])
            grad = grad.at[numeraire_idx].multiply(derivative)
            aux = aux.at[:, numeraire_idx].multiply(derivative)

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
    iterations = state.iter_num.item()  # FIX: Track actual BFGS iterations

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

    if assert_converge:
        assert success, message

    # Hinv relies on the unscaled case-level gradients, which we cleanly preserve
    grad_n = state.aux
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


# EOF
