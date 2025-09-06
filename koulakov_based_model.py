from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
from scipy.spatial.distance import pdist, squareform
from enum import Enum

class Conditions(Enum):
    WT = "WT"
    TEN3_DRG_KO = "Ten3 DRG cKO"
    LPHN2_DRG_KO = "Lphn2 DRG cKO"
    TEN3_DH_KO = "Ten3 DH cKO"
    LPHN2_DH_KO = "Lphn2 DH cKO"


class ActModFunc(Enum):
    HILL = "hill"
    LINEAR = "linear"


@dataclass
class Axon:
    idx: int
    pd_mod_origin: Tuple[float, float]  # (X, Y) in [0,1]×[0,1], proximal-distal and modality origin
    synapses: np.ndarray[np.int_]              # indices of dendrites this axon contacts


@dataclass
class Dendrite:
    idx: int
    dh_pos: Tuple[float, float]      # (x, y) in [0,1]×[0,1]
    synapses: np.ndarray[np.int_]              # indices of axons making contact


@staticmethod
def hill_like(x, a, b, c = 1.0):
    """ Based on Hill equation (with 1/b = hill coefficient, c = k_d).
        Parameter a controls the steepness of the curve and b controls the shape.
        Returns a value between 0 and 1.
    """
    if a == 0:
        if type(x) == np.ndarray:
            return np.zeros(x.shape)
        else:
            return 0
    else:
        ans = a * (x**(1/b) / (x**(1/b) + c))
        # since it can be negative (shouldn't be > 1 though so that part isn't really needed)
        if type(ans) == np.ndarray:
            return np.clip(ans, 0, 1)  
        else:
            return max(0, min(ans, 1))


@staticmethod
def linear(x, a):
    return np.clip(a * x, 0, 1)


class DH_ST_Model:
    """Based on the model in Triplett et al. 2011.

    Parameters follow the notation of the Supplemental Methods: gamma, a, b for the
    activity term; A, B, D for competition; alpha, beta for chemoaffinity, etc.
    """

    def __init__(
        self,
        dh_dims: Tuple[int, int] = (20, 10),
        # Activity‑dependent parameters (Eq. 1.2)
        gamma: float = 0.05,  # overall strength of activity term
        a_act: float = 3.0,   
        b_act: float = 11.0, 
        # Competition parameters (Eqs. 1.4‑1.5)
        A_comp: float = 500.0,  # Axon term (promotes more synapses)
        B_comp: float = 1.0,    # limits synapses per axon
        D_comp: float = 1.0,    # limits synapses per dendrite
        # Chemoaffinity parameters (Eq. 1.11)
        alpha_ca: float = 20.0,  # strength of A system (Ten3-Lphn2 and Ten3-Ten3)
        beta_ca: float = 0.0,   # stength of B system (not used) 
        # Additional binding and rate constants
        k_tt: float = 1.0,  # strength of Ten3 -> Ten3 signalling
        k_lt: float = 1.0,  # strength of Lphn2 -> Ten3 signalling
        k_tl: float = 1.0,  # strength of Lphn2 -> Ten3 signalling
        K_tl: float = 1.0,  # binding constant for Ten3-Lphn2
        K_tt: float = 1.0,  # binding constant for Ten3-Ten3
        # Constants determining how the activity energy should change in KO conditions
        a_t_drg: float = 0.0,  # slope for loss of Tenm3 in drg
        b_t_drg: float = 1.0,  # hill coefficient for loss of Tenm3 in drg
        a_l_drg: float = 0.0,  # slope for loss of Lphn2 in drg
        b_l_drg: float = 1.0,  # hill coefficient for loss of Lphn2 in drg
        a_t_dh: float = 0.0,   # slope for loss of Tenm3 in dh
        b_t_dh: float = 1.0,   # hill coefficient for loss of Tenm3 in dh
        a_l_dh: float = 0.0,   # slope for loss of Lphn2 in dh
        b_l_dh: float = 1.0,   # hill coefficient for loss of Lphn2 in dh
        # Misc simulation parameters
        max_synapses_per_axon: int = 500,
        steps_per_axon: int = 500,
        rng_seed: int | None = 1234,
        condition: Conditions = Conditions.WT,
        act_mod_func: ActModFunc = ActModFunc.HILL
    ) -> None:
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.dh_dims = dh_dims
        self.drg_dims = dh_dims 
        self.steps_per_axon = steps_per_axon
        self.max_synapses_per_axon = max_synapses_per_axon

        # Model parameters
        # From original paper
        self.gamma = gamma
        self.a_act = a_act
        self.b_act = b_act
        self.A_comp = A_comp
        self.B_comp = B_comp
        self.D_comp = D_comp
        self.alpha_ca = alpha_ca
        self.beta_ca = beta_ca
        # Additional binding and rate constants
        self.k_tt = k_tt
        self.k_lt = k_lt
        self.k_tl = k_tl
        self.K_tl = K_tl
        self.K_tt = K_tt
        # For how loss of Tenm3 and Lphn2 affect the activity energy
        self.a_t_drg = a_t_drg
        self.b_t_drg = b_t_drg
        self.a_l_drg = a_l_drg
        self.b_l_drg = b_l_drg
        self.a_t_dh = a_t_dh
        self.b_t_dh = b_t_dh
        self.a_l_dh = a_l_dh
        self.b_l_dh = b_l_dh

        # Build retina & SC lattices 
        self.axons = self._generate_drg()
        self.dendrites = self._generate_dh()

        # Convenience vectors for # synapses per axon / dendrite (mainly for competition energy)
        # NOTE: not simply counts of all synapses but per axon/dendrite so can just sum across all to get total # synapses
        self.axon_syn = np.zeros(len(self.axons), dtype=int)
        self.dend_syn = np.zeros(len(self.dendrites), dtype=int)

        # synapse pairs cache for activity energy calculation
        self.synapse_pairs = np.empty(shape=(0, 2), dtype=int)

        # Pre‑compute DRG & DH coordinate arrays for speed 
        self._axon_coords = np.array([ax.pd_mod_origin for ax in self.axons])
        self._dend_coords = np.array([d.dh_pos for d in self.dendrites])

        # Pre-compute receptor and ligand levels for all positions
        def normalize_coords(pos: Tuple[float, float], dims: Tuple[int, int]) -> Tuple[float, float]:
            if dims[0] == 1:
                x = 0
            else:
                x = pos[0] / (dims[0] - 1)
            if dims[1] == 1:
                y = 0
            else:
                y = pos[1] / (dims[1] - 1)
            return (x, y)

        self._drg_Lphn2_levels = np.array([self.drg_Lphn2_levels(*normalize_coords(pos, self.drg_dims)) for pos in self._axon_coords])
        self._drg_Tenm3_levels = np.array([self.drg_Tenm3_levels(*normalize_coords(pos, self.drg_dims)) for pos in self._axon_coords])
        self._drg_B_levels = np.array([self.drg_B_levels(*normalize_coords(pos, self.drg_dims)) for pos in self._axon_coords])

        self._dh_Lphn2_levels = np.array([self.dh_Lphn2_levels(*normalize_coords(pos, self.dh_dims)) for pos in self._dend_coords])
        self._dh_Tenm3_levels = np.array([self.dh_Tenm3_levels(*normalize_coords(pos, self.dh_dims)) for pos in self._dend_coords])
        self._dh_B_levels = np.array([self.dh_B_levels(*normalize_coords(pos, self.dh_dims)) for pos in self._dend_coords])
        
        self._delta_drg_Lphn2 = np.zeros(self._drg_Lphn2_levels.shape)
        self._delta_drg_Tenm3 = np.zeros(self._drg_Tenm3_levels.shape)
        self._delta_drg_B = np.zeros(self._drg_B_levels.shape)

        self._delta_dh_Lphn2 = np.zeros(self._dh_Lphn2_levels.shape)
        self._delta_dh_Tenm3 = np.zeros(self._dh_Tenm3_levels.shape)
        self._delta_dh_B = np.zeros(self._dh_B_levels.shape)
        
        if condition == Conditions.TEN3_DRG_KO:
            self._delta_drg_Tenm3 = self._drg_Tenm3_levels  # should technically be abs(-drg_Tenm3_levels) but works out to the same thing
            self._drg_Tenm3_levels = np.zeros(self._drg_Tenm3_levels.shape)
        elif condition == Conditions.LPHN2_DRG_KO:
            self._delta_drg_Lphn2 = self._drg_Lphn2_levels
            self._drg_Lphn2_levels = np.zeros(self._drg_Lphn2_levels.shape)
        elif condition == Conditions.TEN3_DH_KO:
            self._delta_dh_Tenm3 = self._dh_Tenm3_levels
            self._dh_Tenm3_levels = np.zeros(self._dh_Tenm3_levels.shape)
        elif condition == Conditions.LPHN2_DH_KO:
            self._delta_dh_Lphn2 = self._dh_Lphn2_levels
            self._dh_Lphn2_levels = np.zeros(self._dh_Lphn2_levels.shape)

        # just pre-compute this here since it will always be the same throughout the run
        if act_mod_func == ActModFunc.HILL:
            self.delta_a_mod = {
                "Tenm3_drg": 1 - hill_like(self._delta_drg_Tenm3, self.a_t_drg, self.b_t_drg),
                "Lphn2_drg": 1 - hill_like(self._delta_drg_Lphn2, self.a_l_drg, self.b_l_drg),
                "Tenm3_dh": 1 - hill_like(self._delta_dh_Tenm3, self.a_t_dh, self.b_t_dh),
                "Lphn2_dh": 1 - hill_like(self._delta_dh_Lphn2, self.a_l_dh, self.b_l_dh)
            }
        elif act_mod_func == ActModFunc.LINEAR:
            self.delta_a_mod = {
                "Tenm3_drg": 1 - linear(self._delta_drg_Tenm3, self.a_t_drg),
                "Lphn2_drg": 1 - linear(self._delta_drg_Lphn2, self.a_l_drg),
                "Tenm3_dh": 1 - linear(self._delta_dh_Tenm3, self.a_t_dh),
                "Lphn2_dh": 1 - linear(self._delta_dh_Lphn2, self.a_l_dh)
            }
        else:
            raise ValueError(f"Invalid activation model function: {act_mod_func}")

        # Pre-compute and cache the pairwise distances between all DH coordinates
        coords = self._dend_coords
        self._dh_dists = squareform(pdist(coords, metric='euclidean'))

        # Pre-compute and cache the pairwise distances between all DRG coordinates
        coords = self._axon_coords
        self._drg_dists = squareform(pdist(coords, metric='euclidean'))

        # Energy bookkeeping 
        self.E_activity = 0
        self.E_competition = 0
        self.E_chemoaffinity = 0
        self.total_energy = 0
        
        # Summary stats for plotting 
        self.energy_history = {
                            "total": [self.total_energy],
                            "activity": [self.E_activity], 
                            "competition": [self.E_competition],
                            "chemoaffinity": [self.E_chemoaffinity]
                            }
        self.n_synapse_history = [sum(self.axon_syn)]

        self.condition = condition
        self.labels = {}

        self.n_create_accepted = 0
        self.n_create_rejected = 0
        self.n_eliminate_accepted = 0
        self.n_eliminate_rejected = 0


    # gradient helper functions
    @staticmethod
    def drg_Lphn2_levels(X: float, Y: float) -> Tuple[float, float]:
        """ Note: X and Y must be in [0, 1]×[0, 1]
        """
        return math.exp(-X)


    @staticmethod
    def drg_Tenm3_levels(X: float, Y: float) -> Tuple[float, float]:
        return math.exp(-(1.0 - X))


    @staticmethod
    def drg_B_levels(X: float, Y: float) -> Tuple[float, float]:
        return math.exp(-Y)
    

    @staticmethod
    def dh_Lphn2_levels(x: float, y: float) -> Tuple[float, float]:
        return math.exp(-x)


    @staticmethod
    def dh_Tenm3_levels(x: float, y: float) -> Tuple[float, float]:
        return math.exp(-(1.0 - x))


    @staticmethod
    def dh_B_levels(x: float, y: float) -> Tuple[float, float]:
        return math.exp(-y)


    # synapse management helpers
    def _add_synapse(self, ax_idx: int | np.int_, dend_idx: int | np.int_) -> None:
        self.axons[ax_idx].synapses = np.append(self.axons[ax_idx].synapses, dend_idx)
        self.dendrites[dend_idx].synapses = np.append(self.dendrites[dend_idx].synapses, ax_idx)
        self.axon_syn[ax_idx] += 1
        self.dend_syn[dend_idx] += 1

        # Update synapse pairs cache
        self.synapse_pairs = np.vstack((self.synapse_pairs, np.array([ax_idx, dend_idx])))


    def _remove_synapse(self, ax_idx: int | np.int_, dend_idx: int | np.int_) -> None:
        syn_idx = np.where(self.axons[ax_idx].synapses == dend_idx)[0][0]
        self.axons[ax_idx].synapses = np.delete(self.axons[ax_idx].synapses, syn_idx)
        syn_idx = np.where(self.dendrites[dend_idx].synapses == ax_idx)[0][0]
        self.dendrites[dend_idx].synapses = np.delete(self.dendrites[dend_idx].synapses, syn_idx)
        self.axon_syn[ax_idx] -= 1
        self.dend_syn[dend_idx] -= 1

        # Update synapse pairs cache
        pair_index = np.where((self.synapse_pairs[:, 0] == ax_idx) & (self.synapse_pairs[:, 1] == dend_idx))[0][0]
        self.synapse_pairs = np.delete(self.synapse_pairs, pair_index, axis=0)


    # energy calculation helpers
    def _compute_activity_energy_component(self, dh_distances: np.ndarray, drg_distances: np.ndarray) -> float:
        """ Compute the activity energy component for a given set of distances.
        """
        C = np.exp(-np.abs(drg_distances) / self.b_act)
        U = np.exp(-(dh_distances ** 2) / (2 * self.a_act ** 2))
        # Sum all contributions        
        E = -(self.gamma / 2) * np.sum(C * U)
        return float(E)


    def _compute_competition_energy_component(self, axon_synapse_counts: np.ndarray | int | float, dendrite_synapse_counts: np.ndarray | int | float) -> float:
        """ Compute the competition energy component for a given set of synapse counts.
        """
        
        # Axon terms (eqs 1.3 + 1.5)
        ax_term = -self.A_comp * (axon_synapse_counts ** (1/2)) + self.B_comp * (axon_synapse_counts ** 2)
        # Dendrite term (eqs 1.3 + 1.4)
        dend_term = self.D_comp * (dendrite_synapse_counts ** 2)

        return float(np.sum(ax_term) + np.sum(dend_term))


    def _compute_chemoaffinity_energy_component(self, ax_idx: int | np.int_, dend_idx: int | np.int_) -> float:
        """ Compute the chemoaffinity energy component for a specific synapse between an axon and a dendrite.
        """
        drg_Lphn2 = self._drg_Lphn2_levels[ax_idx]
        drg_Tenm3 = self._drg_Tenm3_levels[ax_idx]
        drg_B = self._drg_B_levels[ax_idx]

        dh_Lphn2 = self._dh_Lphn2_levels[dend_idx]
        dh_Tenm3 = self._dh_Tenm3_levels[dend_idx]
        dh_B = self._dh_B_levels[dend_idx]
        
        tl_repulsion = dh_Tenm3 * drg_Lphn2 * self.K_tl * self.k_tl
        lt_repulsion = dh_Lphn2 * drg_Tenm3 * self.K_tl * self.k_lt
        repulsion = self.alpha_ca * (lt_repulsion + tl_repulsion)
                
        b_system_attraction = drg_B * dh_B
                
        tt_attraction = drg_Tenm3 * dh_Tenm3 * self.K_tt * self.k_tt
        attraction = (self.beta_ca * b_system_attraction) + (self.alpha_ca * tt_attraction)
                
        return repulsion - attraction


    def _compute_change_activity_energy(self, ax_idx: int | np.int_, dend_idx: int | np.int_, remove: bool = False) -> float:
        """ Compute the change in activity energy for adding a specific synapse between an axon and a dendrite.
            If remove is True, compute the change in energy for removing the synapse instead.
        """

        if self.gamma == 0.0 or self.synapse_pairs.shape[0] == 0:
            return 0.0
        
        # get the indices of the axons and dendrites making up the synapses
        ax_indices = self.synapse_pairs[:, 0].astype(np.int_)
        dend_indices = self.synapse_pairs[:, 1].astype(np.int_)

        # Get the distances between this synapse and all existing synapses
        dh_distances = self._dh_dists[dend_idx, :][dend_indices]
        drg_distances = self._drg_dists[ax_idx, :][ax_indices]
        E = self._compute_activity_energy_component(dh_distances, drg_distances)

        E *= self.delta_a_mod["Tenm3_drg"][ax_idx] * self.delta_a_mod["Lphn2_drg"][ax_idx]
        E *= self.delta_a_mod["Tenm3_dh"][dend_idx] * self.delta_a_mod["Lphn2_dh"][dend_idx]

        if remove:
            return -E
        else:
            return E


    def _compute_change_competition_energy(self, ax_idx: int | np.int_, dend_idx: int | np.int_, remove: bool = False) -> float:
        """ Compute the change in competition energy for adding a specific synapse between an axon and a dendrite.
            If remove is True, compute the change in energy for removing the synapse instead. 

            This is more complex than the activity energy because the competition energy is a function of the square of the number of synapses per axon/dendrite,
            so the change in energy depends on the number of synapses already on the axon/dendrite.

            Fortunately, still a sum over each axon/dendrite so can just calculate the change in the affected axon/dendrite and add it to the total.
        """
        # This can maybe be made more efficient but going with this for now
        # Get the current energy contribution of the affected axon/dendrite
        E_before = self._compute_competition_energy_component(self.axon_syn[ax_idx], self.dend_syn[dend_idx])

        # Get the new number of synapses on the affected axon/dendrite
        new_ax_syn = self.axon_syn[ax_idx] + 1 if not remove else self.axon_syn[ax_idx] - 1
        new_dend_syn = self.dend_syn[dend_idx] + 1 if not remove else self.dend_syn[dend_idx] - 1

        # Get the change in energy
        dE = self._compute_competition_energy_component(new_ax_syn, new_dend_syn) - E_before

        return float(dE)


    def _compute_change_chemoaffinity_energy(self, ax_idx: int | np.int_, dend_idx: int | np.int_, remove: bool = False) -> float:
        """ Compute the change in chemoaffinity energy for adding a specific synapse between an axon and a dendrite.
            If remove is True, compute the change in energy for removing the synapse instead.
        """
        if remove:
            return -self._compute_chemoaffinity_energy_component(ax_idx, dend_idx)
        else:
            return self._compute_chemoaffinity_energy_component(ax_idx, dend_idx)


    # drg and dh generation helpers
    def _generate_drg(self) -> np.ndarray[Axon]:    
        w, h = self.drg_dims
        drg = np.empty(w * h, dtype=object)
        idx = 0
        for x in range(w):
            for y in range(h):
                drg[idx] = Axon(idx=idx, pd_mod_origin=(x, y), synapses=np.array([], dtype=int))
                idx += 1
        return drg


    def _generate_dh(self) -> np.ndarray[Dendrite]:
        w, h = self.dh_dims
        dh = np.empty(w * h, dtype=object)
        idx = 0
        for x in range(w):
            for y in range(h):
                dh[idx] = Dendrite(idx=idx, dh_pos=(x, y), synapses=np.array([], dtype=int))
                idx += 1
        return dh


    # main simulation loop
    def step(self) -> None:
        # Each step, attempt one create and one eliminate
        self._attempt_create()
        self._attempt_eliminate()
        # Update the energy and synapse number histories
        self.energy_history["total"].append(self.total_energy)
        self.energy_history["activity"].append(self.E_activity)
        self.energy_history["competition"].append(self.E_competition)
        self.energy_history["chemoaffinity"].append(self.E_chemoaffinity)
        # sum across array of synapses per axon to get total # synapses
        self.n_synapse_history.append(sum(self.axon_syn))    


    def run(self, verbose: bool = True) -> bool:
        total_attempts = self.steps_per_axon * len(self.axons)
        for step in range(total_attempts):
            self.step()
            # Print the energy every 1000 steps
            if verbose and step % 10000 == 0:
                print(f"Step {step} of {total_attempts}: Energy = {self.total_energy}, Synapses = {sum(self.axon_syn)}")

            if len(self.synapse_pairs) > self.max_synapses_per_axon * len(self.axons):
                if verbose: 
                    print(f"Max synapses per axon surpassed at step {step} of {total_attempts}")
                return False

        return True


    # synapse management helpers
    def _attempt_create(self) -> bool:
        ax_idx = self.rng.randrange(len(self.axons))
        dend_idx = self.rng.randrange(len(self.dendrites))
        dE_act = self._compute_change_activity_energy(ax_idx, dend_idx, remove=False)
        dE_comp = self._compute_change_competition_energy(ax_idx, dend_idx, remove=False)
        dE_ca = self._compute_change_chemoaffinity_energy(ax_idx, dend_idx, remove=False)
        dE = dE_act + dE_comp + dE_ca
        if self._accept_move(dE):
            self._add_synapse(ax_idx, dend_idx)
            self.total_energy += dE
            self.E_activity += dE_act
            self.E_competition += dE_comp
            self.E_chemoaffinity += dE_ca
            self.n_create_accepted += 1
            return True
        else:
            self.n_create_rejected += 1
            return False


    def _attempt_eliminate(self) -> bool:
        # Pick a random existing synapse
        axon_indices = np.where(self.axon_syn > 0)[0]
        nonempty_axons = self.axons[axon_indices]
        if len(nonempty_axons) == 0:
            return False
        ax = self.rng.choice(nonempty_axons)
        dend_idx = self.rng.choice(ax.synapses)
        dE_act = self._compute_change_activity_energy(ax.idx, dend_idx, remove=True)
        dE_comp = self._compute_change_competition_energy(ax.idx, dend_idx, remove=True)
        dE_ca = self._compute_change_chemoaffinity_energy(ax.idx, dend_idx, remove=True)
        dE = dE_act + dE_comp + dE_ca
        if self._accept_move(dE):
            self._remove_synapse(ax.idx, dend_idx)
            self.total_energy += dE
            self.E_activity += dE_act
            self.E_competition += dE_comp
            self.E_chemoaffinity += dE_ca
            self.n_eliminate_accepted += 1
            return True
        else:
            self.n_eliminate_rejected += 1
            return False


    def _accept_move(self, dE: float) -> bool:
        # Avoid overflow by handling large values directly
        if dE > 100:  # If energy increase is very large, always reject
            return False
        elif dE < -100:  # If energy decrease is very large, always accept
            return True
        else:
            P = 1.0 / (1.0 + math.exp(4.0 * dE))
            return random.random() < P


    # analysis helper functions
    def axon_centroids(self):
        """Return SC centroid for each axon (NaN if axon has no synapses)."""
        centroids = np.full((len(self.axons), 2), np.nan)
        for ax in self.axons:
            if len(ax.synapses) > 0:
                coords = self._dend_coords[ax.synapses]
                centroids[ax.idx] = coords.mean(axis=0)
        return centroids

   
    def add_label(self, label_name: str, position_criteria: Callable[[float, float], bool | float]) -> None:
        """ Add a label to the model. If the label is already present, add the new label to the existing
            label instead of replacing it. If position_criteria returns a bool, the label is 1 if the axon's 
            ST origin is within the position_criteria, and 0 otherwise. If position_criteria returns a float, 
            the label is the float returned by position_criteria.
        """
        labels = np.zeros(len(self.axons), dtype=float)
        for ax in self.axons:
            res = position_criteria(*ax.pd_mod_origin)
            if isinstance(res, bool) and res:
                labels[ax.idx] = 1
            else:
                labels[ax.idx] = res    
        if label_name in self.labels:
            self.labels[label_name] = self.labels[label_name] + labels
        else:
            self.labels[label_name] = labels


    def remove_label(self, label_name: str) -> None:
        """ Remove a label from the model.
        """
        self.labels.pop(label_name)


    def get_label_profile_drg(self, label_name: str, axis: int = 0, bin: bool = True, n_bins: int = 10) -> np.ndarray:
        """ Get the label profile for a given label in the DRG/periphery.
            If bin is True, the label profile is the proportion of synapses with the specified label in each bin.
            If bin is False, the label profile is the proportion of synapses with the specified label at each coordinate.

            Returns a 1D array of length n_bins if bin is True, or the same length as the axis if bin is False.
        """        
        # sum the number of labelled synapses in each bin
        label = self.labels[label_name]
        drg_label = np.zeros(self.drg_dims[axis])
        for ax in self.axons:
            drg_label[ax.pd_mod_origin[axis]] += label[ax.idx]
        if bin:
            bin_counts = np.zeros(n_bins)
            for i in range(n_bins):
                start = i*len(drg_label)
                end = (i+1)*len(drg_label)
                bin_counts[i] = np.sum(drg_label[start:end]) 
        else:
            bin_counts = drg_label

        # normalize the bin counts
        # set all values less than 1e-10 to 0
        bin_counts[bin_counts < 1e-10] = 0
        if np.sum(bin_counts) > 0:
            bin_counts = (bin_counts / np.sum(bin_counts)) * 100

        return bin_counts


    def get_label_profile_dh(self, label_name: str, axis: int = 0, bin: bool = True, n_bins: int = 10) -> np.ndarray:
        """ Get the label profile for a given label in the DH.
        """
        label = self.labels[label_name]
        dh_label = np.zeros(self.dh_dims[axis])
        for ax in self.axons:
            for dend_idx in ax.synapses:
                dh_label[self.dendrites[dend_idx].dh_pos[axis]] += label[ax.idx]
        
        if bin:
            bin_counts = np.zeros(n_bins)
            for i in range(n_bins):
                start = int(i*(len(dh_label)/n_bins))
                end = int((i+1)*(len(dh_label)/n_bins))
                bin_counts[i] = np.sum(dh_label[start:end])
        else:
            bin_counts = dh_label
        
        # normalize the bin counts
        # set all values less than 1e-10 to 0
        bin_counts[bin_counts < 1e-10] = 0
        if np.sum(bin_counts) > 0:
            bin_counts = (bin_counts / np.sum(bin_counts)) * 100

        return bin_counts


    # plotting functions
    def correlation_plots(self, show=True, axis: int = 0, normalize_axes: bool = True, save_path: str = None) -> Tuple[plt.Figure, plt.Axes, float]:
        """ Plots the correlation between the DRG and DH coordinates.
            Also returns the correlation coefficients.
        """

        centroids = self.axon_centroids()
        mask = ~np.isnan(centroids[:, 0])
        x_dh, y_dh = centroids[mask].T
        X_drg, Y_drg = self._axon_coords[mask].T
        axes = [
            (X_drg, x_dh, "DRG (D→P)", "DH (M→L)"),
            (Y_drg, y_dh, "DRG Y", "DH y (S→D)"),
        ]
        
        drg, dh, label_x, label_y = axes[axis]

        label = f"{label_x}  → {label_y}"

        if normalize_axes:
            drg = drg / (self.drg_dims[axis] - 1)
            dh = dh / (self.dh_dims[axis] - 1)

        if len(drg) == 0 or len(dh) == 0:
            print(f"No data to plot for {label}")
            return None, None
            
        fig, ax = plt.subplots()
        ax.scatter(drg, dh, s=12)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(f"Correlation: {label}")
            
        # Calculate correlation coefficient
        corr_coef = np.corrcoef(drg, dh)[0, 1]
        # Add correlation value to the plot
        ax.text(0.05, 0.95, f"r = {corr_coef:.3f}", 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
        # Set limits with a small buffer (5% of range) so points aren't right on the edge
        if normalize_axes:
            x_range = 1.0
            y_range = 1.0
        else:
            x_range = self.drg_dims[axis]
            y_range = self.dh_dims[axis]
        buffer_x = 0.05 * x_range
        buffer_y = 0.05 * y_range
        ax.set_xlim(-buffer_x, x_range + buffer_x)
        ax.set_ylim(-buffer_y, y_range + buffer_y)

        fig.set_size_inches(2.5, 2.5)

        if show:
            plt.show()
        if save_path:
            fig.savefig(save_path)
        return ax, corr_coef


    def energy_history_plot(self, show=True, energy_type: str = "total") -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        ax.plot(self.energy_history[energy_type])
        ax.set_xlabel("Steps")
        ax.set_ylabel("Energy")
        ax.set_title("Energy history")
        if show:
            plt.show()
        return fig, ax


    def n_synapse_history_plot(self, show=True) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        ax.plot(self.n_synapse_history)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Number of synapses")


    def synapse_density_plot(self, axis: int = 0, show=True, n_bins: int = None) -> Tuple[plt.Figure, plt.Axes]:
        """ Plot the synapse density along the specified DH axis.
        """

        if n_bins is None:
            n_bins = round(self.dh_dims[axis] / 5)

        # Get the coordinates of the synapses
        synapse_coords = []
        for ax in self.axons:
            if len(ax.synapses) > 0:
                for dend_idx in ax.synapses:
                    synapse_coords.append(self.dendrites[dend_idx].dh_pos[axis])
        
        # Create a histogram of the synapse coordinates
        fig, ax = plt.subplots()
        ax.hist(synapse_coords, bins=n_bins, edgecolor='black')
        ax.set_xlabel("DH coordinate")
        ax.set_ylabel("Synapse density")
        ax.set_title(f"Synapse density along DH {axis}")
        ax.set_xlim(0, self.dh_dims[axis]-1)
        if show:
            plt.show()
        return fig, ax


    def _plot_label_profile(self, bin_counts: np.ndarray, axis: int = 0, bin: bool = True, show=True) -> Tuple[plt.Figure, plt.Axes]:
        """ Plot the label profile along the specified DH axis.
        """
        
        # plot the bin counts
        fig, ax = plt.subplots()
        ax.plot(bin_counts)
        if bin:
            ax.set_xlabel("Bin index")
        else:
            ax.set_xlabel("DH coordinate")
        ax.set_ylabel("Number of labelled synapses")
        ax.set_title(f"Label profile along DH {axis}")
        if show:
            plt.show()
        return fig, ax


    def plot_label_profile_drg(self, label_name: str, axis: int = 0, bin: bool = True, n_bins: int = 10, show=True) -> Tuple[plt.Figure, plt.Axes]:
        bin_counts = self.get_label_profile_drg(label_name, axis, bin, n_bins)
        return self._plot_label_profile(bin_counts, axis, bin, show)


    def plot_label_profile_dh(self, label_name: str, axis: int = 0, bin: bool = True, n_bins: int = 10, show=True) -> Tuple[plt.Figure, plt.Axes]:
        bin_counts = self.get_label_profile_dh(label_name, axis, bin, n_bins)
        return self._plot_label_profile(bin_counts, axis, bin, show)


    def plot_E_act_mod(self, show=True) -> plt.Figure:
        d_Ten3 = np.linspace(0, 1, 100)
        coefficients = [
            (self.a_t_drg, self.b_t_drg, "Ten3 DRG"),
            (self.a_l_drg, self.b_l_drg, "Lphn2 DRG"),
            (self.a_t_dh, self.b_t_dh, "Ten3 DH"),
            (self.a_l_dh, self.b_l_dh, "Lphn2 DH")
        ]
        for a, b, label in coefficients:
            E_act_mod = 1 - hill_like(d_Ten3, a, b)
            plt.plot(d_Ten3, E_act_mod, label=label)
        plt.legend()
        plt.xlabel("Change in expression")
        plt.ylabel("E_act_mod")
        if show:
            plt.show()
        return plt.subplots()


if __name__ == "__main__":
    model = DH_ST_Model(
            dh_dims=(50, 1),
            steps_per_axon=500
            )
    print(f"Number of axons: {len(model.axons)}; dendrites: {len(model.dendrites)}")

    model.run()
    model.correlation_plots()
