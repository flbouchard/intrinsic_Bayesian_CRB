# -*- coding: utf-8 -*
##############################################################################
# A file for testing the Bayes CRB for covariance matrix estimation when
# data follow a complex Gaussian distribution with an inverse Wishart prior
# Authored by Guillaume Ginolhac, 06/09/2024
# e-mail: guillaume.ginolhac@univ-smb.fr
##############################################################################

import sys
import numpy as np
import scipy as sp
import warnings
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
import tikzplotlib
import warnings
from crb_computation import CRB_Covar_Gaussian_Complex, CRB_Covar_Gaussian_IW_Complex\
    ,ICRB_Covar_Gaussian, ICRB_Covar_Gaussian_IW_Complex
from basis import generate_covariance_iw_complex, generate_data_for_estimation_complex\
    , SCM_MAP_IW_COMPLEX, SCM_MMSE_IW_COMPLEX, SCM, dist_euclidian, dist_HPD, ToeplitzMatrix
warnings.simplefilter('once', UserWarning)

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Monte - Carlo
# ---------------------------------------------------------------------------------------------------------------

def one_monte_carlo(trial_no,p, n, mu, Sigma_0_comp, nu):
    """ Fonction for computing MSE - one monte carlo
        ----------------------------------------------------------------------------------
    Inputs:
        p : data size
        n : number of samples
        mu : mean 
        Sigma_0_comp : scale matrix of the inverse Wishart distribution
        nu : degrees of freedom of the inverse Wishart distribution
    
    Outputs :
        Squared Error (eculidean and natural) for MLE, MAP, MMSE"""

    np.random.seed(trial_no)

    # COMPLEX CASE
    Sigma_IW_comp = generate_covariance_iw_complex(p, nu, mu, Sigma_0_comp,0)
    X_comp = generate_data_for_estimation_complex(p, n, mu, Sigma_IW_comp,0)
    Sigma_MAP_IW_comp = SCM_MAP_IW_COMPLEX(X_comp, Sigma_0_comp,nu)
    Sigma_MMSE_IW_comp = SCM_MMSE_IW_COMPLEX(X_comp, Sigma_0_comp,nu)
    
    X_comp = generate_data_for_estimation_complex(p, n, mu, Sigma_0_comp,0)
    Sigma_MLE_comp = SCM(X_comp)    
        
    d_euclidian_Sigma_MAP_IW_comp = dist_euclidian(Sigma_IW_comp, Sigma_MAP_IW_comp)**2 # Natural distance to true value, only shape matrix
    d_euclidian_Sigma_MMSE_IW_comp = dist_euclidian(Sigma_IW_comp, Sigma_MMSE_IW_comp)**2 # Natural distance to true value, only shape matrix
    d_euclidian_Sigma_MLE_comp = dist_euclidian(Sigma_0_comp, Sigma_MLE_comp)**2 # Natural distance to true value, only shape matrix
    d_natural_Sigma_MAP_IW = dist_HPD(Sigma_IW_comp, Sigma_MAP_IW_comp)**2 # Natural distance to true value, only shape matrix
    d_natural_Sigma_MMSE_IW = dist_HPD(Sigma_IW_comp, Sigma_MMSE_IW_comp)**2 # Natural distance to true value, only shape matrix
    d_natural_Sigma_MLE = dist_HPD(Sigma_0_comp, Sigma_MLE_comp)**2 # Natural distance to true value, only shape matrix
    
    return [d_euclidian_Sigma_MAP_IW_comp,d_euclidian_Sigma_MMSE_IW_comp,d_euclidian_Sigma_MLE_comp,d_natural_Sigma_MAP_IW,d_natural_Sigma_MMSE_IW,d_natural_Sigma_MLE]
            
def parallel_monte_carlo(p, n, mu, Sigma_0_comp, nu, number_of_threads, number_of_trials, Multi):
    """ Fonction for computing MSE - mutliple parallel (or not) monte carlo
        ----------------------------------------------------------------------------------
    Inputs:
        p : data size
        n : number of samples
        mu : mean 
        Sigma_0_comp : scale matrix of the inverse Wishart distribution
        nu : degrees of freedom of the inverse Wishart distribution
        number_of_threads : number of threads
        number_of_trials : number of monte carlo
        multi : True for parallel computing, False for sequential
    
    Outputs :
        MSE (eculidean and natural) for MLE, MAP, MMSE"""


    if Multi:
        results_parallel = Parallel(n_jobs=number_of_threads)(delayed(one_monte_carlo)(iMC,p, n, mu, Sigma_0_comp, nu) for iMC in range(number_of_trials))
        results_parallel = np.array(results_parallel)
        d_euclidian_Sigma_MAP_IW = np.mean(results_parallel[:,0], axis=0)
        d_euclidian_Sigma_MMSE_IW = np.mean(results_parallel[:,1], axis=0)
        d_euclidian_Sigma_MLE = np.mean(results_parallel[:,2], axis=0)
        d_natural_Sigma_MAP_IW = np.mean(results_parallel[:,3], axis=0)
        d_natural_Sigma_MMSE_IW = np.mean(results_parallel[:,4], axis=0)
        d_natural_Sigma_MLE = np.mean(results_parallel[:,5], axis=0)
        return d_euclidian_Sigma_MAP_IW, d_euclidian_Sigma_MMSE_IW, d_euclidian_Sigma_MLE, d_natural_Sigma_MAP_IW, d_natural_Sigma_MMSE_IW, d_natural_Sigma_MLE
    else:
        # Results container
        results = []
        for iMC in range(number_of_trials):
            results.append(one_monte_carlo(iMC,p, n, mu, Sigma_0_comp, nu))

        results = np.array(results)
        d_euclidian_Sigma_MAP_IW = np.mean(results[:,0], axis=0)
        d_euclidian_Sigma_MMSE_IW = np.mean(results[:,1], axis=0)
        d_euclidian_Sigma_MLE = np.mean(results[:,2], axis=0)
        d_natural_Sigma_MAP_IW = np.mean(results[:,3], axis=0)
        d_natural_Sigma_MMSE_IW = np.mean(results[:,4], axis=0)
        d_natural_Sigma_MLE = np.mean(results[:,5], axis=0)
        return d_euclidian_Sigma_MAP_IW, d_euclidian_Sigma_MMSE_IW, d_euclidian_Sigma_MLE, d_natural_Sigma_MAP_IW, d_natural_Sigma_MMSE_IW, d_natural_Sigma_MLE
    
# ---------------------------------------------------------------------------------------------------------------
# Definition of Program Principal
# ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    # ---------------------------------------------------------------------------------------------------------------
    # Simulation Parameters
    # ---------------------------------------------------------------------------------------------------------------
    number_of_threads = -1
    Multi = False
    p = 5                                                       # Dimension of data
    n_vec = np.unique(np.logspace(1.01,3.5,15).astype(int))     # Number of samples
    number_of_trials = 100                                      # Number of trials for each point of the MSE
    mu = np.zeros(p)                                             # Mean of Gaussian distribution
    rho_comp = (1/np.sqrt(2))*(0.5+1j*0.5)                        # Toeplitz coefficient for shape matrix
    nu=100                                                       # degrees of freedom of the inverse Wishart distribution
    Sigma_0_comp = ToeplitzMatrix(rho_comp, p)                        # Toeplitz scale matrix of the inverse Wishart distribution
    
    # ---------------------------------------------------------------------------------------------------------------
    #  CRB Computation
    # ---------------------------------------------------------------------------------------------------------------

    CRB_Gaussian=CRB_Covar_Gaussian_Complex(Sigma_0_comp,n_vec)  
    CRB_Gaussian_IW,CRB_Gaussian_IW_asymptotic=CRB_Covar_Gaussian_IW_Complex(Sigma_0_comp,nu,n_vec)
    ICRB_Gaussian=ICRB_Covar_Gaussian(Sigma_0_comp,1,n_vec)
    ICRB_Gaussian_IW,ICRB_Gaussian_IW_asymptotic=ICRB_Covar_Gaussian_IW_Complex(Sigma_0_comp,nu,n_vec)

    # ---------------------------------------------------------------------------------------------------------------
    # MSE computations
    # ---------------------------------------------------------------------------------------------------------------

    print(u"Parameters: p=%d, n=%s, rho=%.2f+1j%.2f" % (p,n_vec,rho_comp.real,rho_comp.imag))
    t_beginning = time.time()

    # Distance containers
    d_euclidian_Sigma_MAP_IW = np.zeros(len(n_vec))
    d_euclidian_Sigma_MMSE_IW = np.zeros(len(n_vec))
    d_euclidian_Sigma_MLE = np.zeros(len(n_vec))
    d_natural_Sigma_MAP_IW = np.zeros(len(n_vec))
    d_natural_Sigma_MMSE_IW = np.zeros(len(n_vec))
    d_natural_Sigma_MLE = np.zeros(len(n_vec))

    for i_n, n in enumerate(tqdm(n_vec)):
        d_euclidian_Sigma_MAP_IW[i_n],d_euclidian_Sigma_MMSE_IW[i_n],d_euclidian_Sigma_MLE[i_n]\
            , d_natural_Sigma_MAP_IW[i_n],d_natural_Sigma_MMSE_IW[i_n],d_natural_Sigma_MLE[i_n] = \
            np.array(parallel_monte_carlo(p, n_vec[i_n], mu, Sigma_0_comp, nu, number_of_threads\
                                          , number_of_trials, Multi))

    print('Done in %f s'%(time.time()-t_beginning))

    # ---------------------------------------------------------------------------------------------------------------
    # Plotting using Matplotlib
    # ---------------------------------------------------------------------------------------------------------------
    markers = ['o', 's' , '*', '8', 'P', 'D', 'X']

    
    plt.figure()
    plt.loglog(n_vec, d_euclidian_Sigma_MLE, marker=markers[0], label='SCM ')
    plt.loglog(n_vec, CRB_Gaussian, marker=markers[2], label='CRB')
    plt.xlabel(r'$n$')
    plt.ylabel(r'MSE $\left( \mathbf{\Sigma}, \hat{\mathbf{\Sigma}} \right)$')
    plt.legend()
    plt.title(r"Euclidean CRB. Parameter: $p=%d$" %p)
    tikzplotlib.save('Euclidean CRB.tex')
    
    plt.figure()
    plt.loglog(n_vec, d_euclidian_Sigma_MAP_IW, marker=markers[0], label='MAP')
    plt.loglog(n_vec, d_euclidian_Sigma_MMSE_IW, marker=markers[1], label='MMSE')
    plt.loglog(n_vec, CRB_Gaussian_IW, marker=markers[2], label='BCRB')
    plt.loglog(n_vec, CRB_Gaussian_IW_asymptotic, marker=markers[3], label='BCRB-Asymptotic')
    plt.xlabel(r'$n$')
    plt.ylabel(r'MSE $\left( \mathbf{\Sigma}, \hat{\mathbf{\Sigma}} \right)$')
    plt.legend()
    plt.title(r"Bayesian Euclidean CRB. Parameters: $p=%d$, $\nu=%d$" % (p,nu))
    tikzplotlib.save(('Euclidean Bayesian CRB ' + str(nu) + '.tex'))
    
    plt.figure()
    plt.loglog(n_vec, d_natural_Sigma_MLE, marker=markers[0], label='SCM')
    plt.loglog(n_vec, ICRB_Gaussian, marker=markers[2], label='ICRB')
    plt.xlabel(r'$n$')
    plt.ylabel(r'MSE $\left( \mathbf{\Sigma}, \hat{\mathbf{\Sigma}} \right)$')
    plt.legend()
    plt.title(r"Intrinsic CRB. Parameter: $p=%d$" %p)
    tikzplotlib.save('Intrinsic CRB.tex')

    plt.figure()
    plt.loglog(n_vec, d_natural_Sigma_MAP_IW, marker=markers[0], label='MAP')
    plt.loglog(n_vec, d_natural_Sigma_MMSE_IW, marker=markers[1], label='MMSE')
    plt.loglog(n_vec, ICRB_Gaussian_IW, marker=markers[2], label='BICRB') 
    plt.loglog(n_vec, ICRB_Gaussian_IW_asymptotic, marker=markers[3], label='BICRB-Asymptotic') 
    plt.xlabel(r'$n$')
    plt.ylabel(r'MSE $\left( \mathbf{\Sigma}, \hat{\mathbf{\Sigma}} \right)$')
    plt.legend()
    plt.title(r"Bayesian Intrinsic CRB. Parameters: $p=%d$, $\nu=%d$" % (p,nu))
    tikzplotlib.save(('Intrinsic Bayesian CRB ' + str(nu) + '.tex'))

    plt.show()
