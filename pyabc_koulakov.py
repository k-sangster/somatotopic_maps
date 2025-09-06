import os
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable
import pyabc
from koulakov_based_model import DH_ST_Model, Conditions
from koulakov_experiments import score_koulakov_model_mse, run_all_conditions, compare_label_profiles, find_label_placement
from data import OBSERVED_PROFILES, PRIORS_WT, PRIORS_ALL_CONDITIONS

# constants
DH_DIMS = (50, 1)
STEPS_PER_AXON = 500
WT_MSE_THRESHOLD = 0.3
INVALID_OUTPUT_PENALTY = 10000
ALPHA = 0.1 # for the quantile epsilon
EPSILON = pyabc.epsilon.QuantileEpsilon(alpha=ALPHA)
MAX_NR_POPULATIONS = 10
POPULATION_SIZE = 1000
N_REPEATS = 1


def model_WT(parameters: dict[str, float]) -> dict[str, float]:
    """ Model wrapper for pyabc

        parameters: dict[str, float]
            Parameters for the model.

        Returns:
            dict[str, float]
                Data dict that is sent to the distance function.
    """
    dh_model = DH_ST_Model(
            dh_dims=DH_DIMS,
            init_synapses_per_axon = 0,
            beta_ca = 0.0,
            steps_per_axon=STEPS_PER_AXON,
            rng_seed=1234,
            **parameters
            )
    valid =  dh_model.run(verbose=False)
    if valid:
        score = score_koulakov_model_mse(dh_model)
    else:
        score = INVALID_OUTPUT_PENALTY
    return {"score": score}  # data dict that is sent to the distance function


def model_all_conditions(parameters: dict[str, float], n_repeats: int = N_REPEATS, WT_mse_threshold: float = WT_MSE_THRESHOLD) -> dict[str, float]:
    """ Model wrapper for pyabc

        parameters: dict[str, float]
            Parameters for the model.

        Returns:
            dict[str, float]
                Data dict that is sent to the distance function.
    """

    parameters = parameters.copy()

    # modify parameters to pass to run_all_conditions
    parameters["dh_dims"] = DH_DIMS
    parameters["steps_per_axon"] = STEPS_PER_AXON  # expected by run_all_conditions, but fixed for pyabc runs
    parameters["beta_ca"] = 0.0

    # do one WT run, check if it is within WT_mse_threshold
    single_wt_model = run_all_conditions(parameters, n_repeats=1, conditions=[Conditions.WT])[Conditions.WT][0]
    if single_wt_model is None:
        return {"valid": False}
    WT_score = score_koulakov_model_mse(single_wt_model)
    if WT_score > WT_mse_threshold:
        return {"valid": False}

    # find the optimal label placement based on the WT profile using one example WT model 
    distal_placements = find_label_placement(single_wt_model, "Distal", OBSERVED_PROFILES["WT_Distal_profile"])
    proximal_placement = find_label_placement(single_wt_model, "Proximal", OBSERVED_PROFILES["WT_Proximal_profile"])[0]
    label_placements = {"Distal": distal_placements, "Proximal": proximal_placement}

    # finish rest of WT models
    if n_repeats > 1:
        models_WT = run_all_conditions(parameters, n_repeats=n_repeats-1, conditions=[Conditions.WT], label_placements=label_placements)
        if models_WT is None:
            return {"valid": False}
        models_WT[Conditions.WT].append(single_wt_model) # add the one we already did
    else:
        models_WT = {Conditions.WT: [single_wt_model]}

    avg_n_synapse_WT = np.mean([len(model.synapse_pairs) for model in models_WT[Conditions.WT]])

    # run the cKO models
    models_cKO = run_all_conditions(parameters, 
                                    n_repeats=n_repeats, 
                                    conditions=[Conditions.TEN3_DH_KO, Conditions.TEN3_DRG_KO, Conditions.LPHN2_DH_KO, Conditions.LPHN2_DRG_KO],
                                    label_placements=label_placements)
    if models_cKO is None:
        return {"valid": False}
    models = {**models_WT, **models_cKO}

    # compare the profiles
    n_bins = min(DH_DIMS[0], OBSERVED_PROFILES["WT_Distal_profile"].shape[0])
    profiles_distal, avg_pos_distal = compare_label_profiles(models, label_name="Distal", axis=0, bin=True, n_bins=n_bins)
    profiles_proximal, avg_pos_proximal = compare_label_profiles(models, label_name="Proximal", axis=0, bin=True, n_bins=n_bins)

    # check if the models produced valid profiles
    if avg_pos_distal is None or avg_pos_proximal is None:
        return {"valid": False}

    # package everything into a single dict
    output = {"valid": True, 
              "distal_placements": distal_placements, 
              "proximal_placement": proximal_placement,
              "n_synapses_WT": avg_n_synapse_WT}
    for condition in Conditions:  
        output[f"{condition.value}_Distal_avg_pos"] = avg_pos_distal[condition]
        output[f"{condition.value}_Proximal_avg_pos"] = avg_pos_proximal[condition]
        output[f"{condition.value}_Distal_profile"] = profiles_distal[condition]
        output[f"{condition.value}_Proximal_profile"] = profiles_proximal[condition]

    return output


def distance_WT(x: dict[str, float], x0: dict[str, float]) -> float:
    """ When an observed value is provided, this is fed into x0 by providing an 
        initial value in the abc.new function (can ignore in our case). x0 also
        called y in some implementations to reflect this.
    """
    return x["score"]  # temporary for testing (since model is already returning a score [mse] that we can use for distance)


def distance_all_conditions_full(x: dict[str, float], x0: dict[str, float]) -> float:
    """ Directly compare the observed and predicted profiles.
    """
    if not x["valid"]:
        # model produced an invalid profile, so we give it a large penalty
        # (likely due to the cutoff values being too extreme or no synapses
        # from the corresponding DRG neurons)
        return INVALID_OUTPUT_PENALTY

    # sanity check on the keys 
    # although not all keys in x will have a corresponding x0 key, all keys in x0 must be in x
    if not all(key in x.keys() for key in x0.keys()):
        missing_keys = set(x0.keys()) - set(x.keys())
        raise ValueError(f"x is missing keys: {missing_keys}")

    mae = 0
    for condition in Conditions:
        for label in ["Distal", "Proximal"]:
            key = f"{condition.value}_{label}_profile"
            # don't have observations for every combination, so only compare the ones that we have data for
            if key in x0.keys():
                # calculate the mse between the observed and predicted profiles
                mae += np.sum(np.abs(x[key] - x0[key]))

    return mae


def pyabc_run(model_fn: Callable, output_dir: str, prior:pyabc.Distribution, distance_fn: Callable, population_size: int, min_epsilon: float, x0: dict | None = None, run_name: str = "", continue_from_db: str | None = None, n_procs: int | None = None):
    """ Run the ABCSMC algorithm.
    """

    # define the ABCSMC algorithm
    abc = pyabc.ABCSMC(
        model_fn,
        prior,
        distance_fn,
        transitions=pyabc.transition.LocalTransition(),
        eps=EPSILON,
        population_size=population_size,
        sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_procs)
    )

    if continue_from_db is not None:
        print(f"Continuing from db: {continue_from_db}")
        abc.load("sqlite:///" + continue_from_db)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(output_dir, f"pyabc_run_{run_name}_{timestamp}.db")
    print(f"Saving db to {db_path}")

    if x0 is not None:
        abc.new("sqlite:///" + db_path, x0)
    else:
        abc.new("sqlite:///" + db_path)

    history = abc.run(minimum_epsilon=min_epsilon, max_nr_populations=MAX_NR_POPULATIONS)

    # save the results
    with open(os.path.join(output_dir, f"pyabc_history_{run_name}_{timestamp}.pkl"), "wb") as f:
        pickle.dump(history, f)

    with open(os.path.join(output_dir, f"pyabc_abc_object_{run_name}_{timestamp}.pkl"), "wb") as f:
        pickle.dump(abc, f)


def pyabc_run_WT(output_dir: str, continue_from_db: str | None = None, custom_prior_db: str | None = None, additional_run_suffix: str = "", constrained_comp_prior: bool = False, n_procs: int | None = None) -> None:
    """ Run the ABCSMC algorithm for the WT model.
    """

    if custom_prior_db is None:
        prior = pyabc.Distribution(**PRIORS_WT)
    else:
        prior = CustomPrior(custom_prior_db, **PRIORS_WT)

    if constrained_comp_prior:
        prior = CompetitionConstrainedPrior(num_axons=DH_DIMS[0]*DH_DIMS[1], num_dendrites=DH_DIMS[0]*DH_DIMS[1], **PRIORS_WT)

    additional_run_suffix = "_constrained_comp" if constrained_comp_prior else ""
    additional_run_suffix += "_" + additional_run_suffix if additional_run_suffix else ""
    population_size = pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=10000, mean_cv = 0.1)
    pyabc_run(model_WT, output_dir, prior, distance_WT, population_size=population_size, min_epsilon=0.001, run_name="WT" + additional_run_suffix, continue_from_db=continue_from_db, n_procs=n_procs)


def pyabc_run_all_conditions(output_dir: str, continue_from_db: str | None = None, custom_prior_db: str | None = None, additional_run_suffix: str = "", constrained_comp_prior: bool = False, n_procs: int | None = None) -> None:
    """ Run the ABCSMC algorithm for all conditions.
    """

    if custom_prior_db is None:
        prior = pyabc.Distribution(**PRIORS_ALL_CONDITIONS)
    else:
        prior = CustomPrior(custom_prior_db, **PRIORS_ALL_CONDITIONS)

    if constrained_comp_prior:
        prior = CompetitionConstrainedPrior(num_axons=DH_DIMS[0]*DH_DIMS[1], num_dendrites=DH_DIMS[0]*DH_DIMS[1], **PRIORS_ALL_CONDITIONS)

    x0 = OBSERVED_PROFILES

    min_epsilon = 0.005  # 0.01
    suffix = "_constrained_comp" if constrained_comp_prior else ""
    suffix += "_" + additional_run_suffix if additional_run_suffix else ""
    pyabc_run(model_all_conditions, output_dir, prior, distance_all_conditions_full, population_size=POPULATION_SIZE, min_epsilon=min_epsilon, x0=x0, run_name="all_conditions" + suffix, continue_from_db=continue_from_db, n_procs=n_procs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyABC parameter inference for the Koulakov model")
    parser.add_argument("--mode", choices=["WT", "all"], default="all", 
                        help="Run inference for WT only or all conditions (default: all)")
    parser.add_argument("--continue_from", type=str, default=None,   # TODO: try to use this to continue the all conditions run from the WT run
                        help="Path to the database file to continue from")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="Suffix to add to the run name")
    parser.add_argument("--db_prior", type=str, default=None,
                        help="Path to the database file to use as a custom prior")
    parser.add_argument("--constrained_comp_prior", action="store_true",
                        help="Use a competition-constrained prior")
    parser.add_argument("--n_procs", type=int, default=None,
                        help="Number of processes to use for the sampler. If None, will be determined automatically.")
    parser.add_argument("--output_dir", type=str, default="./pyabc_runs",
                        help="Path to the output directory")
    args = parser.parse_args()
    
    if args.mode == "WT":
        pyabc_run_WT(args.output_dir, continue_from_db=args.continue_from, custom_prior_db=args.db_prior, additional_run_suffix=args.run_suffix, n_procs=args.n_procs)
    elif args.mode == "all":
        pyabc_run_all_conditions(args.output_dir, continue_from_db=args.continue_from, custom_prior_db=args.db_prior, additional_run_suffix=args.run_suffix, n_procs=args.n_procs)
    else:
        print(f"Unknown mode: {args.mode}. Using default (all conditions).")
        pyabc_run_all_conditions(args.output_dir, continue_from_db=args.continue_from, custom_prior_db=args.db_prior, additional_run_suffix=args.run_suffix, n_procs=args.n_procs)
    
