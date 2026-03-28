"""Estimation for latent-class conditional logit."""

from dataclasses import replace
from time import time
from typing import Sequence

from jax.typing import ArrayLike

from lcl._case_utils import _diff_unchosen_chosen, _to_structural_betas
from lcl._choice_model import ChoiceModel
from lcl._em_alg_startup import _get_starting_vals
from lcl._em_alg_steps import _em_alg, _update_betas
from lcl._results import LCLResults
from lcl._struct import EMAlgConfig, MleConfig


class LatentClassConditionalLogit(ChoiceModel):
    """Specification and estimation for latent-class conditional logit models."""

    def __init__(
        self,
        num_classes: int = 5,
        numeraire: str | None = None,
        numeraire_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.numeraire = numeraire
        self.numeraire_idx = numeraire_idx

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        case_varnames: Sequence[str],
        alts: ArrayLike,
        cases: ArrayLike,
        panels: ArrayLike,
        dems: ArrayLike | None = None,
        dem_varnames: Sequence[str] | None = None,
        em_alg_config: EMAlgConfig = EMAlgConfig(),
        mle_config: MleConfig = MleConfig(),
    ) -> LCLResults:
        """Fit the latent-class conditional logit model."""

        self._pre_fit(case_varnames, dem_varnames, self.numeraire)
        self.num_vars = len(case_varnames)
        self.num_dem_vars = len(dem_varnames) if dem_varnames is not None else 0

        if self.numeraire:
            self.numeraire_idx = self.case_varnames.index(self.numeraire)
        else:
            self.numeraire_idx = None

        data, *_ = self._setup_data(
            X=X, dems=dems, y=y, cases=cases, panels=panels, alts=alts
        )
        diff_unchosen_chosen = _diff_unchosen_chosen(data)

        em_vars = _get_starting_vals(
            diff_unchosen_chosen,
            data,
            self.num_classes,
            em_alg_config,
            mle_config,
            self.numeraire_idx,
        )

        logliks_list, em_recursion = [], 0
        while em_recursion < em_alg_config.maxiter:
            print(f"EM recursion: {em_recursion}")

            em_vars = _em_alg(
                em_vars,
                diff_unchosen_chosen,
                data,
                self.num_classes,
                mle_config,
                self.numeraire_idx,
            )

            logliks_list.append(em_vars.unconditional_loglik)
            em_recursion += 1

            if em_recursion >= 5:
                rel_change = abs(em_vars.unconditional_loglik - logliks_list[-5]) / abs(
                    logliks_list[-5]
                )
                if rel_change <= em_alg_config.loglik_tol:
                    break

        # Run one final EM recursion with ultra-strict tolerances
        strict_mle_config = replace(mle_config, ftol=1e-8, maxiter=500)
        em_vars = _em_alg(
            em_vars,
            diff_unchosen_chosen,
            data,
            self.num_classes,
            strict_mle_config,
            self.numeraire_idx,
        )

        estim_time_sec = time() - self._fit_start_time

        # Pass state to Results object
        return LCLResults(
            model_spec=self,
            em_vars=em_vars,
            estimation_data=data,
            em_recursion=em_recursion,
            em_alg_config=em_alg_config,
            estim_time_sec=estim_time_sec,
        )


# EOF
