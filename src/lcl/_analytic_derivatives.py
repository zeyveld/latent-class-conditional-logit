r"""Analytic score and observed-information derivatives for LCL inference.

Notation. For panel ``n`` and class ``c`` let ``a_nc = log pi_nc(theta) +
kappa_nc(beta_c)``, where ``pi`` is the demographic class-membership prior and
``kappa_nc`` is the panel's conditional-logit log likelihood under class ``c``.
The observed-data panel log likelihood is ``l_n = logsumexp_c a_nc`` with
posterior ``h_nc = softmax_c(a_nc)``.  Writing ``g_nc`` for the (block-sparse)
gradient of ``a_nc``, the exact derivatives of ``l_n`` are

.. math::

    \\nabla l_n = \\sum_c h_{nc} g_{nc}

    \\nabla^2 l_n = \\sum_c h_{nc} [\\nabla^2 a_{nc} + g_{nc} g_{nc}']
                    - (\\nabla l_n)(\\nabla l_n)'

The first identity is the Fisher identity underlying EM (Dempster, Laird &
Rubin, 1977, JRSS-B 39(1):1-38).  The second is the finite-mixture form of the
observed information matrix (Louis, 1982, JRSS-B 44(2):226-233; Oakes, 1999,
JRSS-B 61(2):479-482; McLachlan & Krishnan, 2008, *The EM Algorithm and
Extensions*, 2nd ed., ch. 4).  The class-conditional blocks are the standard
conditional-logit score and Hessian (McFadden, 1974; Train, 2009, *Discrete
Choice Methods with Simulation*, ch. 3) evaluated on the
chosen-alternative-differenced design.

Everything is returned in the flat latent parameter layout owned by
:class:`~lcl._params.ParamPacking`, with the numeraire softplus chain rule
applied exactly as in :mod:`lcl.constraints`.  The outputs are numerically
equal (to machine precision) to ``jax.jacfwd`` of
:meth:`~lcl._results.LCLResults._panel_loglik_fn` and ``jax.hessian`` of
:meth:`~lcl._results.LCLResults._full_loglik_fn`, but require one pass over
the data instead of one pass per parameter.  Apart from the required
``(panels, parameters)`` score and ``(parameters, parameters)`` Hessian outputs,
the data-sized temporaries therefore avoid the extra parameter axis created by
automatic differentiation.
"""

import jax.numpy as jnp
from equinox import filter_jit
from jax.nn import log_softmax, sigmoid, softmax
from jax.ops import segment_max, segment_sum
from jaxtyping import Array, Float64

from lcl._params import ParamPacking
from lcl._struct import Data, DiffUnchosenChosen


@filter_jit
def _panel_scores_and_hessian(
    flat_params: Float64[Array, "all_params"],
    diff_unchosen_chosen: DiffUnchosenChosen,
    data: Data,
    packing: ParamPacking,
) -> tuple[
    Float64[Array, "panels all_params"],
    Float64[Array, "all_params all_params"],
]:
    """Compute panel-level scores and the observed-information Hessian.

    Parameters
    ----------
    flat_params : Float64[Array, "all_params"]
        Latent parameters in the canonical :class:`~lcl._params.ParamPacking`
        layout.
    diff_unchosen_chosen : :class:`~lcl._struct.DiffUnchosenChosen`
        Differenced design matrix.
    data : :class:`~lcl._struct.Data`
        Core choice data and metadata.
    packing : :class:`~lcl._params.ParamPacking`
        Owner of the flat parameter layout and structural transforms.

    Returns
    -------
    panel_scores : Float64[Array, "panels all_params"]
        Per-panel gradients of the observed-data log likelihood; row ``n``
        equals row ``n`` of the Jacobian of
        :meth:`~lcl._results.LCLResults._panel_loglik_fn`.
    hessian : Float64[Array, "all_params all_params"]
        Hessian of the total observed-data log likelihood, equal to
        ``jax.hessian`` of :meth:`~lcl._results.LCLResults._full_loglik_fn`.
    """
    if data.panels_of_cases is None or data.num_panels is None:
        raise ValueError("Panel identifiers are required for LCL derivatives.")

    num_alt_vars = packing.num_alt_vars
    num_classes = packing.num_classes
    theta_rows, theta_cols = packing.theta_shape
    num_beta_params = packing.num_beta_params
    num_theta_params = theta_rows * theta_cols
    num_panels = data.num_panels

    latent_betas, thetas = packing.unpack(flat_params)
    betas = packing.to_structural(latent_betas)

    Xd = diff_unchosen_chosen.X
    cases_d = diff_unchosen_chosen.cases
    num_cases = diff_unchosen_chosen.num_cases
    panels_of_cases = data.panels_of_cases

    # Class-conditional logit pieces on the differenced design, all classes at
    # once (vectorized _diff_logit_components).
    Vd = Xd @ betas
    shift = jnp.maximum(0.0, segment_max(Vd, cases_d, num_segments=num_cases))
    e_shifted = jnp.exp(Vd - shift[cases_d])
    den_shifted = jnp.exp(-shift) + segment_sum(
        e_shifted, cases_d, num_segments=num_cases
    )
    log_chosen = -shift - jnp.log(den_shifted)  # (cases, classes)
    q_rows = e_shifted / den_shifted[cases_d]  # (diff rows, classes)

    kappa = segment_sum(log_chosen, panels_of_cases, num_segments=num_panels)

    # Case-level conditional-logit score of the chosen log probability with
    # respect to structural beta_c: -(sum_j q_ij d_ij).
    xbar = segment_sum(
        Xd[:, :, None] * q_rows[:, None, :], cases_d, num_segments=num_cases
    )  # (cases, alt_vars, classes)
    panel_beta_scores = -segment_sum(
        xbar, panels_of_cases, num_segments=num_panels
    )  # (panels, alt_vars, classes)

    # Membership prior, posterior, and their score pieces.
    if data.dems is None:
        dem_design = jnp.ones((num_panels, 1))
    else:
        dem_design = jnp.concatenate(
            [jnp.ones((data.dems.shape[0], 1), dtype=data.dems.dtype), data.dems],
            axis=1,
        )
    logits = jnp.concatenate([jnp.zeros((num_panels, 1)), dem_design @ thetas], axis=1)
    pi = softmax(logits, axis=1)
    posterior = softmax(log_softmax(logits, axis=1) + kappa, axis=1)

    weighted_beta_scores = posterior[:, None, :] * panel_beta_scores
    pi_tail = pi[:, 1:]
    posterior_tail = posterior[:, 1:]
    theta_scores = dem_design[:, :, None] * (posterior_tail - pi_tail)[:, None, :]

    # Chain rule to latent space for the numeraire row.
    numeraire_idx = packing.numeraire_idx
    if numeraire_idx is not None:
        d1 = -sigmoid(latent_betas[numeraire_idx, :])
        weighted_beta_scores_latent = weighted_beta_scores.at[
            :, numeraire_idx, :
        ].multiply(d1[None, :])
    else:
        weighted_beta_scores_latent = weighted_beta_scores

    panel_scores = jnp.concatenate(
        [
            weighted_beta_scores_latent.reshape(num_panels, num_beta_params),
            theta_scores.reshape(num_panels, num_theta_params),
        ],
        axis=1,
    )

    # ----- Hessian, assembled in structural space ---------------------------
    posterior_by_case = posterior[panels_of_cases]  # (cases, classes)
    posterior_by_row = posterior_by_case[cases_d]  # (diff rows, classes)

    # sum_n sum_c h_nc hess kappa_nc: class-diagonal conditional-logit blocks,
    # hess kappa_c = -(Xd' diag(q_c w_c) Xd - sum_i w_ic xbar_ic xbar_ic').
    # Per-class matmuls keep the peak temporary at one (rows, alt_vars) copy
    # instead of the (rows, alt_vars, classes) intermediate an einsum builds.
    row_w = q_rows * posterior_by_row  # (diff rows, classes)
    second_moment = jnp.stack(
        [(Xd * row_w[:, c][:, None]).T @ Xd for c in range(num_classes)]
    )
    first_moment = jnp.stack(
        [
            (xbar[:, :, c] * posterior_by_case[:, c][:, None]).T @ xbar[:, :, c]
            for c in range(num_classes)
        ]
    )
    cl_blocks = -(second_moment - first_moment)  # (classes, K, K)

    # sum_n sum_c h_nc (beta score)(beta score)': also class-diagonal.
    score_outer = jnp.einsum(
        "nc,nkc,nlc->ckl", posterior, panel_beta_scores, panel_beta_scores
    )

    beta_beta = jnp.zeros((num_alt_vars, num_classes, num_alt_vars, num_classes))
    diag_classes = jnp.arange(num_classes)
    beta_beta = beta_beta.at[:, diag_classes, :, diag_classes].add(
        cl_blocks + score_outer
    )

    # Cross beta-theta blocks: sum_n h_nc s_nc (z_n (delta_{c,m+1} - pi_n,m+1))'.
    cross_delta = jnp.einsum("nkc,nj->kcj", weighted_beta_scores, dem_design)
    beta_theta = -jnp.einsum(
        "nkc,nj,nm->kcjm", weighted_beta_scores, dem_design, pi_tail
    )
    class_tail = jnp.arange(1, num_classes)
    beta_theta = beta_theta.at[:, class_tail, :, class_tail - 1].add(
        jnp.moveaxis(cross_delta[:, 1:, :], 1, 0)
    )

    # Theta-theta blocks: zz' (x) [(diag(h) - h pi' - pi h' + pi pi')
    #                              - (diag(pi) - pi pi')] on non-baseline coords.
    eye_tail = jnp.eye(num_classes - 1)
    membership_curvature = (
        posterior_tail[:, :, None] * eye_tail[None, :, :]
        - posterior_tail[:, :, None] * pi_tail[:, None, :]
        - pi_tail[:, :, None] * posterior_tail[:, None, :]
        + 2.0 * pi_tail[:, :, None] * pi_tail[:, None, :]
        - pi_tail[:, :, None] * eye_tail[None, :, :]
    )
    theta_theta = jnp.einsum(
        "nj,ni,nml->jmil", dem_design, dem_design, membership_curvature
    )

    structural_scores = jnp.concatenate(
        [
            weighted_beta_scores.reshape(num_panels, num_beta_params),
            theta_scores.reshape(num_panels, num_theta_params),
        ],
        axis=1,
    )

    num_params = packing.num_params
    hessian = jnp.zeros((num_params, num_params))
    hessian = hessian.at[:num_beta_params, :num_beta_params].set(
        beta_beta.reshape(num_beta_params, num_beta_params)
    )
    cross_flat = beta_theta.reshape(num_beta_params, num_theta_params)
    hessian = hessian.at[:num_beta_params, num_beta_params:].set(cross_flat)
    hessian = hessian.at[num_beta_params:, :num_beta_params].set(cross_flat.T)
    hessian = hessian.at[num_beta_params:, num_beta_params:].set(
        theta_theta.reshape(num_theta_params, num_theta_params)
    )
    hessian = hessian - structural_scores.T @ structural_scores

    # Pull the Hessian back to latent space through the numeraire transform.
    if numeraire_idx is not None:
        scale = jnp.ones(num_params)
        constrained = numeraire_idx * num_classes + jnp.arange(num_classes)
        d1 = -sigmoid(latent_betas[numeraire_idx, :])
        scale = scale.at[constrained].set(d1)
        hessian = hessian * scale[:, None] * scale[None, :]
        structural_grad = jnp.sum(weighted_beta_scores, axis=0)
        d2 = d1 * (1.0 + d1)
        hessian = hessian.at[constrained, constrained].add(
            structural_grad[numeraire_idx, :] * d2
        )

    return panel_scores, hessian
