import os
import pickle
from datetime import datetime
import numpy as np
import pyabc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import brute
from skimage.measure import block_reduce
from koulakov_based_model import DH_ST_Model, Conditions
from data import OBSERVED_PROFILES, PRIORS_WT, PRIORS_ALL_CONDITIONS


# scoring functions
def score_koulakov_model_cor(model: DH_ST_Model) -> float:
    centroids = model.axon_centroids()
    mask = ~np.isnan(centroids[:, 0])
    x_dh, y_dh = centroids[mask].T
    X_drg, Y_drg = model._axon_coords[mask].T
    correlations = []

    for (drg, dh) in [
        (X_drg, x_dh),
        (Y_drg, y_dh),
    ]:
        if len(drg) == 0 or len(dh) == 0:
            continue
        correlations.append(pearsonr(drg, dh).statistic)

    if len(correlations) == 0:
        return 0.0
    else:
        return np.mean(correlations)


def score_koulakov_model_mse(model: DH_ST_Model) -> float:
    """ Vs just correlation: 
        A high correlation could exist even with systematic offsets or scaling differences, 
        whereas MSE will directly penalize any deviation from the expected position. 
        The MSE approach is likely more stringent about precise topographic mapping.
    """

    centroids = model.axon_centroids()
    mask = ~np.isnan(centroids[:, 0])
    x_dh, y_dh = centroids[mask].T
    X_drg, Y_drg = model._axon_coords[mask].T
    diffs = []
    for (drg, dh, axis) in [
        (X_drg, x_dh, 0),
        (Y_drg, y_dh, 1),
    ]:
        if len(drg) == 0 or len(dh) == 0:
            continue
        # normalize axes
        if model.drg_dims[axis] == 1:
            drg = 0
        else:
            drg = drg / (model.drg_dims[axis] - 1)
        if model.dh_dims[axis] == 1:
            dh = 0
        else:
            dh = dh / (model.dh_dims[axis] - 1)

        # compare dh centroid to ideal position (matching normalized position in drg)
        # using mse but can try other metrics
        diff = (drg - dh) ** 2
        diffs.append(np.mean(diff))

    if len(diffs) == 0:
        # only occurs if there are no synapses
        return 10000  # arbitrary large score to punish degenerate solutions
    else:
        # better to use mean across the 2 dimensions?
        return np.sum(diffs)


# label placement helpers
def simulate_injection(model: DH_ST_Model, label_name: str, center: float, spread: float, concentration: float, spread_type: str = "gaussian") -> None:
    """ Helper function to add a label to the model. """
    center_abs = center * model.drg_dims[0]
    spread_abs = spread * model.drg_dims[0]

    def gaussian(x, y):
        sigma = spread_abs
        mu = center_abs
        # to avoid dividing by 0
        if sigma == 0:
            out = 0.0
        else:
            out = np.exp(-((x - mu) / sigma) ** 2)
        return concentration * out

    def uniform(x, y):
        return float(concentration * int(x >= center_abs - spread_abs and x <= center_abs + spread_abs))

    if spread_type == "gaussian":
        model.add_label(label_name, gaussian)
    elif spread_type == "uniform":
        model.add_label(label_name, uniform)
    else:
        raise ValueError(f"Spread type {spread_type} not supported")


def score_label_placement(model: DH_ST_Model, injections: list[tuple[float, float, float]], target_profile: np.ndarray, spread_type: str = "gaussian") -> float:
    """ Score the placements of a label based on the target profile. """
    label_name = "label"
    for center, spread, concentration in injections:
        simulate_injection(model, label_name, center,
                           spread, concentration, spread_type)
    n_bins = min(model.dh_dims[0], target_profile.shape[0])
    score = np.sum((model.get_label_profile_dh(
        label_name, n_bins=n_bins) - target_profile) ** 2) / n_bins
    model.remove_label(label_name)
    return score


def find_label_placement(model: DH_ST_Model, label_name: str, target_profile: np.ndarray, spread_type: str = "gaussian", Ns: int = 10) -> list[tuple[float, float, float]]:
    """ Find the optimal placement of labels to match the target profile. 
        (Used for setting the cKO label positions based on the WT profile)

    """
    if label_name == "Proximal":
        concentrations = [1.0]
    elif label_name == "Distal":
        concentrations = [1.0, 2.0]
    else:
        raise ValueError(f"Label name {label_name} not supported")

    # define the objective function

    def objective(x):
        # unpack from np.ndarray, each odd index is center, each even is spread
        centers = []
        spreads = []
        for i in range(len(x)):
            if i % 2 == 0:  # 0, 2, 4, ...
                centers.append(x[i])
            else:  # 1, 3, 5, ...
                spreads.append(x[i])

        score = score_label_placement(model, list(
            zip(centers, spreads, concentrations)), target_profile, spread_type)
        return score

    # try grid search instead using scipy.optimize.brute
    if label_name == "Proximal":
        # center is 0.5-1, spread is 0-0.5
        bounds = ((0.5, 1), (0, 0.5))
    elif label_name == "Distal":
        # for distal, need to actually specify 2 points
        # center is 0-0.5, spread is 0-0.5
        bounds = ((0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5))
    else:
        raise ValueError(f"Label name {label_name} not supported")

    res = brute(objective, bounds, Ns=Ns)

    # convert back from np array to list of tuples
    centers = []
    spreads = []
    for i in range(len(res)):
        if i % 2 == 0:  # 0, 2, 4, ...
            centers.append(res[i])
        else:  # 1, 3, 5, ...
            spreads.append(res[i])

    return list(zip(centers, spreads, concentrations))


def visualize_label_placement(model: DH_ST_Model, placements: list[tuple[float, float, float]], target_profile: np.ndarray, spread_type: str = "gaussian") -> None:
    """ Visualize the placement of a label compared to the target profile. """
    label_name = "label"
    for center, spread, concentration in placements:
        simulate_injection(model, label_name, center,
                           spread, concentration, spread_type)
    fig, ax = model.plot_label_profile_dh(
        label_name, show=False, bin=True, n_bins=len(target_profile))
    ax.plot(target_profile)
    ax.legend(["Label profile", "Target profile"])
    plt.show()
    model.remove_label(label_name)

# experiment helpers
def run_all_conditions(params: dict[str, float], n_repeats: int = 10, label_placements: dict[str, tuple[float, float, float]] | None = None, verbose: bool = False, conditions: list[Conditions] = list(Conditions)) -> dict[Conditions, list[DH_ST_Model]] | None:
    """ If any of the models surpass their max number of synapses per axon, 
        return None.
    """

    models = {}
    for condition in conditions:
        models[condition] = []
        if verbose:
            print(condition.value)
        for i in range(n_repeats):
            model = DH_ST_Model(
                condition=condition,
                **params
            )
            valid = model.run(verbose=False)
            if not valid:
                return None
            models[condition].append(model)

    if label_placements is None:
        # find the optimal label placement based on the WT profile using one example WT model
        wt_model = models[Conditions.WT][0]
        distal_placements = find_label_placement(
            wt_model, "Distal", OBSERVED_PROFILES["WT_Distal_profile"])
        proximal_placement = find_label_placement(
            wt_model, "Proximal", OBSERVED_PROFILES["WT_Proximal_profile"])[0]
    else:
        distal_placements = label_placements["Distal"]
        proximal_placement = label_placements["Proximal"]

    for condition in conditions:
        for model in models[condition]:
            for center, spread, concentration in distal_placements:
                simulate_injection(model, "Distal", center,
                                   spread, concentration)
            simulate_injection(
                model, "Proximal", proximal_placement[0], proximal_placement[1], proximal_placement[2])

    return models


def compare_label_profiles(models: dict[Conditions, list[DH_ST_Model]], label_name: str, axis: int = 0, axis_size: int = None, bin: bool = True, n_bins: int = 10) -> tuple[dict[Conditions, np.ndarray], dict[Conditions, float]]:
    """
        Returns the average label profile and the average bin.
        If bin is True, the average label profile is a 1D array of length n_bins.
        If bin is False, the average label profile is a 1D array of length axis_size
        and instead of the average bin, the average position (in [0,1]) of the labelled synapses is returned.
    """

    # set up the array to store the average label profile
    if bin:
        avg_label_profile = {condition: np.zeros(
            n_bins) for condition in Conditions}
    else:
        if axis_size is None:
            # attempt to infer axis size from the first model(assumes all models have same dh_dims)
            axis_size = models[list(models.keys())[0]][0].dh_dims[axis]
        avg_label_profile = {condition: np.zeros(
            axis_size) for condition in Conditions}
    # average label profiles across all models
    for condition in Conditions:
        for model in models[condition]:
            label_profile = model.get_label_profile_dh(
                label_name, axis=axis, bin=bin, n_bins=n_bins)
            avg_label_profile[condition] += label_profile
    avg_label_profile = {
        condition: avg_label_profile[condition] / len(models[condition]) for condition in Conditions}

    # get the "average bin"
    avg_bin = {}
    for condition in Conditions:
        if np.sum(avg_label_profile[condition]) == 0:
            # maybe too strict, but not going to allow any of the conditions to have 0 labelled synapses
            return None, None
        if bin:
            avg_bin[condition] = np.dot(avg_label_profile[condition], np.arange(
                n_bins)) / np.sum(avg_label_profile[condition])
        else:
            # normalize the average position to [0,1]
            avg_bin[condition] = np.dot(avg_label_profile[condition], np.arange(
                axis_size) / (axis_size - 1)) / np.sum(avg_label_profile[condition])

    return avg_label_profile, avg_bin


def plot_label_profiles(models: dict[Conditions, list[DH_ST_Model]], label_names: list[str] | str | None = None, conditions: list[Conditions] = list(Conditions)) -> None:
    if label_names is None:
        label_names = ["Distal", "Proximal"]
    else:
        label_names = [label_names] if isinstance(
            label_names, str) else label_names

    for label_name in label_names:
        avg_label_profile, avg_bin = compare_label_profiles(
            models, label_name, bin=False)
        # plot all label profiles:
        for condition in conditions:
            plt.plot(avg_label_profile[condition], label=condition)
        plt.title(label_name)
        plt.legend()
        plt.show()


def downsample_profile(profile: np.ndarray, n_bins: int) -> np.ndarray:
    """ Downsample the profile to the given number of bins. """
    if profile.shape[0] == n_bins:
        return profile
    else:
        return block_reduce(profile, block_size=n_bins // profile.shape[0], func=np.mean)


def plot_vs_observed(models: dict[Conditions, list[DH_ST_Model]], observed_profiles: dict[str, np.ndarray] = OBSERVED_PROFILES) -> None:
    """ Plot the simulated label profiles vs the observed label profiles. """
    n_bins = min(models[list(models.keys())[0]][0].dh_dims[0],
                 observed_profiles[f"{Conditions.WT.value}_Distal_profile"].shape[0])
    if n_bins < observed_profiles[f"{Conditions.WT.value}_Distal_profile"].shape[0]:
        # downsample the observed profile
        for key in observed_profiles.keys():
            observed_profiles[key] = downsample_profile(
                observed_profiles[key], n_bins)

    for label_name in ["Distal", "Proximal"]:
        avg_label_profile, avg_bin = compare_label_profiles(
            models, label_name, bin=True, n_bins=n_bins)
        for condition in Conditions:
            key = f"{condition.value}_{label_name}_profile"
            if key in observed_profiles.keys():
                plt.plot(avg_label_profile[condition], label="Simulated")
                plt.plot(observed_profiles[key], label="Observed")
                plt.title(f"{condition.value} {label_name} profile")
                plt.legend()
                plt.show()


def generate_random_models(output_dir: str, n_models: int = 100, dh_dims: tuple[int, int] = (50, 1), steps_per_axon: int = 500, priors: dict[str, pyabc.RV] = PRIORS_WT) -> None:
    """ Generate and save n random WT models based on the priors. For testing only."""
    models = []
    for i in range(n_models):
        params = {name: rv.rvs() for name, rv in priors.items()}
        model = DH_ST_Model(
            condition=Conditions.WT,
            dh_dims=dh_dims,
            steps_per_axon=steps_per_axon,
            beta_ca=0.0,
            **params
        )
        model.run(verbose=False)
        models.append(model)
        if i % 10 == 0:
            print(f"Generated {i}/{n_models} models")

    # save the models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"random_models_{timestamp}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(models, f)
    print(f"Saved {n_models} random models to {output_path}")

# save model outputs
def save_label_profiles(models: dict[Conditions, list[DH_ST_Model]], output_dir: str = "output/label_profiles", n_bins: int = 10) -> None:
    """ Save the label profiles of the models to a csv file. """
    for label_name in ["Distal", "Proximal"]:
        avg_label_profile, avg_bin = compare_label_profiles(
            models, label_name, bin=True, n_bins=n_bins)
        for condition in Conditions:
            key = f"{condition.value}_{label_name}_profile"
            profile = avg_label_profile[condition]
            with open(os.path.join(output_dir, f"{key}.csv"), "w") as f:
                f.write(f"position,value\n")
                for i in range(len(profile)):
                    f.write(f"{i},{profile[i]}\n")


def save_pd_to_ml_mapping(models: list[DH_ST_Model], output_dir: str = "output/mappings/") -> None:
    for i, model in enumerate(models):
        centroids = model.axon_centroids()
        mask = ~np.isnan(centroids[:, 0])
        x_dh, y_dh = centroids[mask].T
        X_drg, Y_drg = model._axon_coords[mask].T
        with open(os.path.join(output_dir, f"mapping_{i}.csv"), "w") as f:
            f.write(f"x_dh,y_dh,X_drg,Y_drg\n")
            for i in range(len(x_dh)):
                f.write(f"{x_dh[i]},{y_dh[i]},{X_drg[i]},{Y_drg[i]}\n")


def save_individual_axon_synapses(models: list[DH_ST_Model], output_dir: str = "output/individual_axon_synapses/") -> None:
    for i, model in enumerate(models):
        with open(os.path.join(output_dir, f"axon_synapses_{i}.csv"), "w") as f:
            f.write(f"axon_id,synapse_x,synapse_y\n")
            for axon in model.axons:
                for dend_idx in axon.synapses:
                    x = model.dendrites[dend_idx].dh_pos[0]
                    y = model.dendrites[dend_idx].dh_pos[1]
                    f.write(f"{axon.idx},{x},{y}\n")
