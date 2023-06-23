## Mimic CXR DSCM
Deep Structural Causam Model (DSCM) [1] implementation and training code for the Mimic CXR dataset of chest x-rays as well as counterfactual generation code.

## Datasets
- dscmchest/mimic.py - implementation of the Mimic CXR dataset [2]

## DSCM implementation and training
- dscmchest/pgm_chest.py - normalising flow mechanisms implementation and training
- dscmchest/vae_chest.py - vae implementation
- dscmchest/train_setup.py - initialisation of all datasets and dataloader necessary for training the DSCM.
- dscmchest/hps.py - hyperparameters

## Generation of counterfactuals
- dscmchest/generate_counterfactuals.py - counterfactual generation functions for Mimic CXR

## References
[1] Nick Pawlowski, Daniel Coelho de Castro, and Ben Glocker. Deep structural causal models for tractable counterfactual inference. Advances in Neural Infor- mation Processing Systems, 33:857–869, 2020 \
[2] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic- cxr, a de-identified publicly available database of chest radiographs with free- text reports. Scientific data, 6(1):317, 2019
