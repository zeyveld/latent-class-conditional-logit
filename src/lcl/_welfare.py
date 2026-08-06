"""Consumer-welfare kernels for anticipated and experienced utility."""

import jax.numpy as jnp
from jax.ops import segment_sum
from jaxtyping import Array, Float64, UInt

from lcl._kernels import _choice_probabilities_and_logsum


def _train_welfare_by_class(
    anticipated_X: Float64[Array, "rows alt_vars"],
    experienced_X: Float64[Array, "rows alt_vars"],
    betas: Float64[Array, "alt_vars classes"],
    cases: UInt[Array, "rows"],
    num_cases: int,
) -> dict[str, Float64[Array, "cases classes"]]:
    """Return Train (2015) welfare components for every case and class.

    Consumers choose using anticipated systematic utility ``W`` but receive
    experienced utility ``U``.  With ``d = U - W``, Train's expected experienced
    surplus in utility units is ``logsum(W) + sum_j P_j(W) d_j``.  Perfect-foresight
    surplus is ``logsum(U)``.

    Parameters
    ----------
    anticipated_X : Array
        Design matrix containing attributes used when the consumer chooses.
    experienced_X : Array
        Row-aligned design matrix containing attributes the consumer experiences.
    betas : Array
        Taste parameters, one column per latent class.
    cases : Array
        Contiguous zero-indexed choice-situation identifiers.
    num_cases : int
        Number of choice situations.

    Returns
    -------
    dict[str, Array]
        Class-specific welfare components in utility units.  ``foreknowledge_loss``
        is reported as the nonnegative loss ``logsum(U) - E[U chosen under W]``;
        Train's signed change is its negative.
    """
    if anticipated_X.shape != experienced_X.shape:
        raise ValueError(
            "anticipated_X and experienced_X must have identical shapes."
        )

    anticipated_probs, anticipated_logsum = _choice_probabilities_and_logsum(
        anticipated_X, betas, cases, num_cases
    )
    _, perfect_foresight_logsum = _choice_probabilities_and_logsum(
        experienced_X, betas, cases, num_cases
    )
    utility_difference = (experienced_X - anticipated_X) @ betas
    experience_effect = segment_sum(
        anticipated_probs * utility_difference,
        cases,
        num_segments=num_cases,
    )
    experienced_surplus = anticipated_logsum + experience_effect
    foreknowledge_loss = perfect_foresight_logsum - experienced_surplus

    return {
        "anticipated_surplus": anticipated_logsum,
        "experience_effect": experience_effect,
        "experienced_surplus": experienced_surplus,
        "perfect_foresight_surplus": perfect_foresight_logsum,
        "foreknowledge_loss": foreknowledge_loss,
    }


def _mix_welfare_components(
    welfare_by_class: dict[str, Float64[Array, "cases classes"]],
    class_probs_by_case: Float64[Array, "cases classes"],
    marginal_utility_income: Float64[Array, "classes"] | None,
) -> dict[str, Float64[Array, "cases"]]:
    """Marginalize class-specific welfare in utils and, when possible, dollars."""
    mixed: dict[str, Float64[Array, "cases"]] = {}
    for name, values in welfare_by_class.items():
        mixed[f"{name}_utils"] = jnp.einsum(
            "nc,nc->n", class_probs_by_case, values
        )
        if marginal_utility_income is not None:
            mixed[f"{name}_dollars"] = jnp.einsum(
                "nc,nc->n",
                class_probs_by_case,
                values / marginal_utility_income[None, :],
            )
    return mixed
