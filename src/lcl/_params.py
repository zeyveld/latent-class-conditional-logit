"""Canonical latent-class parameter packing and coefficient bounds."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float64

from lcl.constraints import DEFAULT_NEGATIVE_MIN_ABS, NegativeCoefficientBound
from lcl._kernels import _class_membership_probs


@dataclass(frozen=True)
class ParamPacking:
    """Own the flat parameter layout used by LCL inference.

    Parameters
    ----------
    num_alt_vars : int
        Number of alternative-specific variables.
    num_classes : int
        Number of latent classes.
    num_dem_vars : int
        Number of panel-level demographic variables.
    numeraire_idx : int | None
        Row constrained to be strictly negative, if any.
    numeraire_min_abs : float, default=1e-5
        Minimum absolute magnitude of the constrained coefficient.

    Notes
    -----
    The flat layout is beta parameters in row-major ``(variables, classes)``
    order followed by non-baseline class-membership logits in row-major
    ``(demographics + intercept, classes - 1)`` order.
    """

    num_alt_vars: int
    num_classes: int
    num_dem_vars: int
    numeraire_idx: int | None
    numeraire_min_abs: float = DEFAULT_NEGATIVE_MIN_ABS

    @property
    def num_beta_params(self) -> int:
        """Return the number of class-specific utility parameters."""
        return self.num_alt_vars * self.num_classes

    @property
    def theta_shape(self) -> tuple[int, int]:
        """Return the canonical class-membership coefficient shape."""
        return (self.num_dem_vars + 1, self.num_classes - 1)

    @property
    def num_params(self) -> int:
        """Return the total length of the flat parameter vector."""
        theta_rows, theta_cols = self.theta_shape
        return self.num_beta_params + theta_rows * theta_cols

    @property
    def negative_bound(self) -> NegativeCoefficientBound:
        """Return the resolved bound shared with the class-specific solvers."""
        return NegativeCoefficientBound(self.numeraire_idx, self.numeraire_min_abs)

    def upper_bounds(self) -> Float64[Array, "all_params"] | None:
        """Return coefficient bounds in the same layout as the flat parameters."""
        return self.negative_bound.upper_bounds(
            jnp.zeros(self.num_params), width=self.num_classes
        )

    def pack(
        self,
        betas: Float64[Array, "alt_vars classes"],
        thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"] | None,
        shares: Float64[Array, "classes"] | None,
    ) -> Float64[Array, "all_params"]:
        """Flatten utility and membership parameters into the canonical layout."""
        expected_beta_shape = (self.num_alt_vars, self.num_classes)
        if betas.shape != expected_beta_shape:
            raise ValueError(
                f"betas has shape {betas.shape}; expected {expected_beta_shape}."
            )

        if thetas is None:
            if self.num_dem_vars:
                raise ValueError(
                    "Class-membership coefficients are required when the model "
                    "contains demographics."
                )
            if shares is None:
                raise ValueError("Class shares are required to pack parameters.")
            if shares.shape != (self.num_classes,):
                raise ValueError(
                    f"shares has shape {shares.shape}; expected {(self.num_classes,)}."
                )
            clipped_shares = jnp.clip(shares, 1e-300)
            normalized_shares = clipped_shares / clipped_shares.sum()
            membership_params = (
                jnp.log(normalized_shares[1:]) - jnp.log(normalized_shares[0])
            )[None, :]
        else:
            if thetas.shape != self.theta_shape:
                raise ValueError(
                    f"thetas has shape {thetas.shape}; expected {self.theta_shape}."
                )
            membership_params = thetas

        return jnp.concatenate([betas.ravel(), membership_params.ravel()])

    def unpack(
        self,
        flat_params: Float64[Array, "all_params"],
    ) -> tuple[
        Float64[Array, "alt_vars classes"],
        Float64[Array, "dem_vars_plus_one classes_minus_one"],
    ]:
        """Reconstruct utility and membership matrices from a flat vector."""
        if flat_params.shape != (self.num_params,):
            raise ValueError(
                f"flat_params has shape {flat_params.shape}; "
                f"expected {(self.num_params,)}."
            )
        betas = flat_params[: self.num_beta_params].reshape(
            self.num_alt_vars, self.num_classes
        )
        thetas = flat_params[self.num_beta_params :].reshape(self.theta_shape)
        return betas, thetas

    def class_probs(
        self,
        thetas: Float64[Array, "dem_vars_plus_one classes_minus_one"],
        dems: Float64[Array, "panels dem_vars"] | None,
        num_panels: int,
    ) -> Float64[Array, "panels classes"]:
        """Compute prior class probabilities with explicit shape validation."""
        if dems is None and thetas.shape[0] != 1:
            raise ValueError(
                "Demographics are required because the fitted membership model "
                f"has {thetas.shape[0] - 1} demographic coefficient rows."
            )
        if dems is not None:
            if dems.shape != (num_panels, self.num_dem_vars):
                raise ValueError(
                    f"dems has shape {dems.shape}; expected "
                    f"{(num_panels, self.num_dem_vars)}."
                )
        return _class_membership_probs(thetas, dems, num_panels)
