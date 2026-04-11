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
    """Specification and estimation for latent-class conditional logit models.

    This class provides the interface for defining and fitting a latent-class
    conditional logit model using an Expectation-Maximization (EM) algorithm. It
    inherits from the abstract base class `ChoiceModel` and manages the data
    ingestion, initialization, and iterative optimization of latent taste
    parameters and class membership probabilities.

    Parameters
    ----------
    num_classes : int, default=5
        The number of discrete latent classes to estimate.
    numeraire : str | None, default=None
        The name of the variable to be used as the numeraire (e.g., price or cost).
        If specified, its taste parameter is mathematically constrained to be
        strictly negative across all latent classes via a softplus transformation
        to ensure theoretically consistent willingness-to-pay calculations.

    Attributes
    ----------
    num_classes : int
        The number of discrete latent classes.
    numeraire : str | None
        The name of the numeraire variable.
    numeraire_idx : int | None
        The column index of the numeraire variable in the expanded design matrix,
        resolved during the `fit` method.
    num_vars : int
        The total number of alternative-specific variables (taste parameters),
        resolved during the `fit` method.
    num_dem_vars : int
        The total number of demographic variables, resolved during the `fit` method.
    """

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
        """Fit the latent-class conditional logit model using an EM algorithm.

        This method ingests raw data, translates it into strictly contiguous,
        zero-indexed JAX arrays (PyTrees), and executes the hardware-accelerated
        EM optimization routine.

        Parameters
        ----------
        data : Any
            The main dataset containing choice situations. Accepts a Polars DataFrame,
            Pandas DataFrame, or dictionary of arrays.
        alts_col : str
            The name of the column identifying specific alternatives within a choice
            situation.
        cases_col : str
            The name of the column grouping observations into distinct choice
            situations.
        panels_col : str
            The name of the column mapping choice situations to specific
            decision-makers (panels).
        formula : str | None, default=None
            An R-style formula string (e.g., "choice ~ price + time | income").
            If provided, overrides `choice_col`, `case_varnames`, and `dem_varnames`.
        choice_col : str | None, default=None
            The name of the boolean or binary column indicating chosen alternatives.
            Required if `formula` is not provided.
        case_varnames : Sequence[str] | None, default=None
            A list of alternative-specific variables to include in the utility
            specification. Required if `formula` is not provided.
        dem_varnames : Sequence[str] | None, default=None
            A list of demographic variables used to predict latent class membership.
        dems_data : Any | None, default=None
            An optional, separate panel-level dataset containing demographics. If
            provided, it will be merged with the main `data` on `panels_col`.
        em_alg_config : :class:`~lcl._struct.EMAlgConfig`, default=EMAlgConfig()
            A dataclass (PyTree) containing configuration options for the overall EM
            algorithm (e.g., maximum iterations, tolerance, hardware distribution).
        mle_config : :class:`~lcl._struct.MleConfig`, default=MleConfig()
            A dataclass (PyTree) containing optimization settings for the M-step's
            internal L-BFGS solver.

        Returns
        -------
        :class:`~lcl._results.LCLResults`
            A container holding the estimated parameters, optimization metadata,
            information criteria, and methods for inference (standard errors,
            predictions).

        Raises
        ------
        ValueError
            If a `numeraire` was specified during class instantiation but cannot be
            found in the expanded design matrix columns.
        """

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
                    f"Hardware Status: Distributing {self.num_classes} classes across {num_devices} GPUs."
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

            # Only force a host sync every `check_interval` steps
            if em_recursion >= 5 and (em_recursion % em_alg_config.check_interval == 0):
                # jax.block_until_ready() forces the sync explicitly here
                current_ll = float(em_vars.unconditional_loglik)
                past_ll = float(logliks_list[-5])

                rel_change = abs(current_ll - past_ll) / abs(past_ll)
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
