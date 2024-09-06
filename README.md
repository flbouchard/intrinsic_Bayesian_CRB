# Intrinsic Bayesian Cramér-Rao bound

This is the code corresponding to the article entitled ["Intrinsic Bayesian Cramér-Rao Bound with an Application to Covariance Matrix Estimation"](https://arxiv.org/abs/2311.04748 "iBCRB") authored by F. Bouchard, A. Renaux, G. Ginolhac and A. Breloy.



This repository contains four python files:
* **script_paper.py**: main file, script to reproduce the results displayed in the paper.
* **data_generation.py**: functions to generate data - random covariance matrices and random samples.
* **crb_computation.py**: functions to compute the various Cramér-Rao bounds considered in the paper.
* **matrix_operators.py**: basic operations on symmetric and Hermitian matrices. Taken from [pyRiemann](https://github.com/alexandrebarachant/pyRiemann/blob/master/pyriemann/utils/base.py "pyRiemann") to avoid dependency.

All credit for this code goes to Guillaume Ginolhac.

