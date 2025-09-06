### Description
Python implementation of the Koulakov-based model described in [1] which is based on the model developed in [2-4], as well as code for ABC-SMC inference of model parameters.


### Requirements:
Conda: https://anaconda.org/anaconda/conda

Tested on Ubuntu 20.04.5 LTS

NOTE: The ABC-SMC procedure is computationally intensive and additional PyABC runs will likely require access to a CPU cluster.

### Installation:

	git clone https://github.com/k-sangster/somatotopic_maps
	cd somatotopic_maps
	conda env create -f environment.yml
	conda activate somatotopic_maps


### Eamples:
To run the model with the default parameters (from the retinocollicular model<sup>[4]</sup>):

	python koulakov_based_model.py

To start another PyABC run:

	python pyabc_koulakov.py --output_dir ./pyabc_runs/

See the demo notebooks for additional examples that make use of the fitted parameter values:
- cKO_demo.ipynb
- disrupted_activity_demo.ipynb
- nerve_cut_demo.ipynb


### References:

[1] Sangster, K. T. et al. Teneurin-3 and latrophilin-2 are required for somatotopic map formation and somatosensory topognosis. 2025.08.13.670179 Preprint at https://doi.org/10.1101/2025.08.13.670179 (2025).

[2] Koulakov, A. A. & Tsigankov, D. N. A stochastic model for retinocollicular map development. BMC Neurosci 5, 30 (2004).

[3] Tsigankov, D. N. & Koulakov, A. A. A unifying model for activity-dependent and activity-independent mechanisms predicts complete structure of topographic maps in ephrin-A deficient mice. J Comput Neurosci 21, 101-114 (2006).

[4] Triplett, J. W. et al. Competition is a driving force in topographic mapping. Proceedings of the National Academy of Sciences 108, 19060-19065 (2011).
