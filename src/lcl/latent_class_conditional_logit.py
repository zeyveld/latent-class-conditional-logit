"""Estimation for latent-class conditional logit."""

from dataclasses import replace
from time import time
from typing import Any, Sequence

from lcl._case_utils import _diff_unchosen_chosen
from lcl._choice_model import ChoiceModel
from lcl._em_alg_startup import _get_starting_vals
from lcl._em_alg_steps import _em_alg
from lcl._results import LCLResults
from lcl._struct import EMAlgConfig, MleConfig


class LatentClassConditionalLogit(ChoiceModel):
    """Specification and estimation for latent-class conditional logit models."""

    def __init__(
        self,
        num_classes: int = 5,
        numeraire: str | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.numeraire = numeraire
        self.numeraire_idx: int | None = None

    def fit(
        self,
        data: Any,
        alts_col: str,
        cases_col: str,
        panels_col: str,
        formula: str | None = None,
        choice_col: str | None = None,
        case_varnames: Sequence[str] | None = None,
        dem_varnames: Sequence[str] | None = None,
        dems_data: Any | None = None,
        em_alg_config: EMAlgConfig = EMAlgConfig(),
        mle_config: MleConfig = MleConfig(),
    ) -> LCLResults:
        """Fit the latent-class conditional logit model using the EM algorithm."""

        parsed_data = self._ingest_data(
            data=data,
            alts_col=alts_col,
            cases_col=cases_col,
            panels_col=panels_col,
            formula=formula,
            choice_col=choice_col,
            case_varnames=case_varnames,
            dem_varnames=dem_varnames,
            dems_data=dems_data,
        )

        self._pre_fit(
            parsed_data.case_varnames, parsed_data.dem_varnames, self.numeraire
        )
        self.num_vars = len(self.case_varnames)
        self.num_dem_vars = len(self.dem_varnames) if self.dem_varnames else 0

        if self.numeraire:
            try:
                self.numeraire_idx = self.case_varnames.index(self.numeraire)
            except ValueError:
                raise ValueError(
                    f"Numeraire '{self.numeraire}' not found in expanded design matrix."
                )
        else:
            self.numeraire_idx = None

        data_struct, weights, init_beta = self._setup_data(parsed_data)
        diff_unchosen_chosen = _diff_unchosen_chosen(data_struct)

        em_vars = _get_starting_vals(
            diff_unchosen_chosen,
            data_struct,
            self.num_classes,
            em_alg_config,
            mle_config,
            self.numeraire_idx,
        )

        num_devices = em_alg_config.num_devices
        if num_devices > 1:
            if self.num_classes % num_devices == 0:
                print(
                    f"Hardware Status: Distributing {self.num_classes} classes evenly across {num_devices} GPUs via shard_map."
                )
            else:
                print(
                    f"Hardware Status: Found {num_devices} GPUs, but {self.num_classes} classes cannot be distributed evenly. Falling back to single-device execution."
                )
        else:
            print("Hardware Status: Running on a single device.")

        logliks_list, em_recursion = [], 0
        while em_recursion < em_alg_config.maxiter:
            print(f"EM recursion: {em_recursion}")

            em_vars = _em_alg(
                em_vars,
                diff_unchosen_chosen,
                data_struct,
                self.num_classes,
                mle_config,
                em_alg_config,
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

        strict_mle_config = replace(mle_config, ftol=1e-8, maxiter=500)
        em_vars = _em_alg(
            em_vars,
            diff_unchosen_chosen,
            data_struct,
            self.num_classes,
            strict_mle_config,
            em_alg_config,
            self.numeraire_idx,
        )

        estim_time_sec = time() - self._fit_start_time

        print(f"Estimation time: {estim_time_sec}")

        return LCLResults(
            model_spec=self,
            em_vars=em_vars,
            estimation_data=data_struct,
            em_recursion=em_recursion,
            em_alg_config=em_alg_config,
            estim_time_sec=estim_time_sec,
        )


# EOF
