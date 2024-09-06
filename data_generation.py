

import sys
import numpy as np
import scipy as sp
import warnings
from matrix_operators import invsqrtm, logm, inv
import warnings
warnings.simplefilter('once', UserWarning)

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Generation of Data
# ---------------------------------------------------------------------------------------------------------------

def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    (p, p) = covariance.shape
    Gamma = covariance
    C = pseudo_covariance

    # Computing elements of matrix Gamma_2r
    Gamma_x = 0.5 * np.real(Gamma + C)
    Gamma_xy = 0.5 * np.imag(-Gamma + C)
    Gamma_yx = 0.5 * np.imag(Gamma + C)
    Gamma_y = 0.5 * np.real(Gamma - C)

    # Matrix Gamma_2r as a block matrix
    Gamma_2r = np.block([[Gamma_x, Gamma_xy], [Gamma_yx, Gamma_y]])

    # Generating the real part and imaginary part
    mu = np.hstack((mean.real, mean.imag))
    v = np.random.multivariate_normal(mu, Gamma_2r, N).T
    X = v[0:p, :]
    Y = v[p:, :]
    return X + 1j * Y

def generate_data_for_estimation_complex(p, N, mu, Sigma, pseudo_Sigma):
    X = np.empty((p,N), dtype=complex)
    X = multivariate_complex_normal_samples(mu, Sigma, N, pseudo_Sigma)
    return X

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Generation of Covariance matrices
# ---------------------------------------------------------------------------------------------------------------

def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def generate_covariance_iw_complex(p, nu, mu, Sigma_bar, pseudo_Sigma):
    X = np.empty((p,nu), dtype=complex)
    X = multivariate_complex_normal_samples(mu, inv(Sigma_bar*(nu-p)), nu, pseudo_Sigma)
    iSigma = (X @ X.conj().T)
    return inv(iSigma)

def generate_covariance_iw_real(p, nu, mu, Sigma_bar):
    X = np.empty((p,nu), dtype=float)
    X = generate_data_for_estimation_real(p, nu, mu, inv(Sigma_bar*(nu-p-1)))
    iSigma = (X @ X.T) 
    return inv(iSigma)

def generate_data_for_estimation_real(p, N, mu, Sigma):
    X = np.empty((p,N), dtype=float)
    X = np.random.multivariate_normal(mu, Sigma, N).T
    return X

def generate_covariance_w_complex(p, nu, mu, Sigma_bar, pseudo_Sigma):
    X = np.empty((p,nu), dtype=complex)
    X = multivariate_complex_normal_samples(mu, Sigma_bar, nu, pseudo_Sigma)
    Sigma = (X @ X.conj().T)
    return Sigma

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Covariance Estimation
# ---------------------------------------------------------------------------------------------------------------

def SCM(x, *args):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""

    (p, N) = x.shape
    return (x @ x.conj().T) / N

def SCM_MAP_IW_REAL(x, R_bar,nu):
    """ A function that computes the MAP for covariance matrix estimation and for IW prior
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
                * R_bar,nu = parameters of the IW
            Outputs:
                * R = the estimate"""

    (p, n) = x.shape
    R = ((nu-p-1.0)*R_bar+x @ x.T)/(nu+p+1.0+n)
    return R

def SCM_MAP_IW_COMPLEX(x, R_bar,nu):
    """ A function that computes the MAP for covariance matrix estimation and for IW prior
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
                * R_bar,nu = parameters of the IW
            Outputs:
                * R = the estimate"""

    (p, n) = x.shape
    R = ((nu-p)*R_bar+x @ x.conj().T)/(nu+n+p)
    return R

def SCM_MMSE_IW_COMPLEX(x, R_bar,nu):
    """ A function that computes the MMSE for covariance matrix estimation and for IW prior
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
                * R_bar,nu = parameters of the IW
            Outputs:
                * R = the estimate"""

    (p, n) = x.shape
    R = ((nu-p)*R_bar+x @ x.conj().T)/(nu+n-p)
    return R


# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Computing Distance
# ---------------------------------------------------------------------------------------------------------------

def dist_HPD(Sigma_0, Sigma_1):
    """ Fonction for computing the Riemannian distance between two HPD matrices
        ------------------------------------------------------------------------
        Inputs:
        --------
            * Sigma_0 = HPD matrix of dimension p
            * Sigma_1 = HPD matrix of dimension p

        Outputs:
        ---------
            * the distance
        """

    isqrtmSigma_0 = invsqrtm(Sigma_0)
    return np.linalg.norm( logm( isqrtmSigma_0 @ Sigma_1 @ isqrtmSigma_0 ), 'fro')

def dist_euclidian(Sigma_0, Sigma_1):
    """ Fonction for computing the euclidian distance between matrices
        ----------------------------------------------------------------------------------
        Inputs:
        --------
            * Sigma_0 = HPD matrix of dimension p
            * Sigma_1 = HPD matrix of dimension p

        Outputs:
        ---------
            * the distance
        """
    return np.linalg.norm(Sigma_0 - Sigma_1, 'fro')

