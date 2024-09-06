
import numpy as np
from matrix_operators import sqrtm, inv
from scipy.stats import ortho_group

#------------------------------------------------------------------------
# Computation of basis of the tangent space
#------------------------------------------------------------------------
def basis_euc_sym_mat_complex(M):
    """ A function that computes the euclidian basis of hermitian matrices.
            Inputs:
                * M : matrix size
            Outputs:
                * the elements of the basis """

    A=np.zeros((M,M,M**2), dtype=complex)
    index=0
    
    # Basis of the diag parts
    for i in range(M):
        A[i,i,index]=1
        index=index+1
    
    # Basis of the real parts    
    for i in range(M):
        for j in range(i+1):
            if (i!=j):
                A[i,j,index]=1/np.sqrt(2)
                A[j,i,index]=1/np.sqrt(2)
                index=index+1
    
    # Basis of the imaginary part
    for i in range(M):
        for j in range(i+1):
            if i!=j:
                A[i,j,index]=1j/np.sqrt(2)
                A[j,i,index]=-1j/np.sqrt(2)
                index=index+1
    return A

def basis_euc_sym_mat_real(M):
    """ A function that computes the euclidian basis of symmetric matrices.
                Inputs:
                    * M : matrix size
                Outputs:
                    * the elements of the basis """
    lindex = int(M*(M+1)/2)
    Omega=np.zeros((M,M,lindex),dtype=float)
    index=0
    # Basis of the diag parts
    for i in range(M):
        Omega[i,i,index]=1
        index=index+1
        
    for i in range(M):
        for j in range(i+1):
            if (i!=j):
                Omega[i,j,index]=1/np.sqrt(2)
                Omega[j,i,index]=1/np.sqrt(2)
                index=index+1
    return Omega

def basis_euc_sym_mat_nat(p,Sigma):
    """ A function that computes the natural basis of hermitian matrices at Sigma
                Inputs:
                    * M : matrix size
                    * Sigma : origin point
                Outputs:
                    * the elements of the basis """

    M = int(p*(p+1)/2)
    Omega_N=np.zeros((p,p,M),dtype=float)
    Omega=basis_euc_sym_mat_real(p)
    U = ortho_group.rvs(p)
    H=sqrtm(Sigma)@U
    
    for i in range(M):
        Omega_N[:,:,i]=H@Omega[:,:,i]@H.T
        
    return Omega_N

#------------------------------------------------------------------------
# Euclidean CRB Without Prior
#------------------------------------------------------------------------

def CRB_Covar_Gaussian_Real(Sigma,scale):
    """ A function that computes the euclidean CRB for real covar matrice estimation
                Inputs:
                    * Sigma : true parameter
                    * scale : array of number of training data for the covar estimation
                Outputs:
                    * the CRB """

    m=Sigma.shape[0]
    N=scale.shape
    Sigma_inv = inv(Sigma)

    # Construction basis
    Omega = basis_euc_sym_mat_real(m)
    M = Omega.shape[2] 

    # Construction de la FIM
    F=np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            F[i,j]=np.matrix.trace(Sigma_inv@Omega[:,:,i]@Sigma_inv@Omega[:,:,j])

    CRB_Gaussian=np.zeros(N)
    CRB_Gaussian=2*np.matrix.trace(inv(F))/scale
    return(CRB_Gaussian)

def CRB_Covar_Gaussian_Complex(Sigma,scale):
    """ A function that computes the euclidean CRB for complex covar matrice estimation
                Inputs:
                    * Sigma : true parameter
                    * scale : array of number of training data for the covar estimation
                Outputs:
                    * the CRB """
    
    m=Sigma.shape[0]
    N=scale.shape
    Sigma_inv = inv(Sigma)

    # Construction basis
    Omega = basis_euc_sym_mat_complex(m)

    # Construction de la FIM
    F=np.zeros((m**2,m**2))
    for i in range(m**2):
        for j in range(m**2):
            F[i,j]=np.matrix.trace(Sigma_inv@Omega[:,:,i]@Sigma_inv@Omega[:,:,j])

    CRB_Gaussian=np.zeros(N)
    CRB_Gaussian=np.matrix.trace(inv(F))/scale
    return(CRB_Gaussian)

#------------------------------------------------------------------------
# Natural CRB Without Prior
#------------------------------------------------------------------------

def ICRB_Covar_Gaussian(Sigma,iComp,scale):
    """ A function that computes the intrinsic CRB for covar matrice estimation
                Inputs:
                    * Sigma : true parameter
                    * iComp : 1 for complex data and 0 for real data
                    * scale : array of number of training data for the covar estimation
                Outputs:
                    * the CRB """
    
    m=Sigma.shape[0]
    if iComp == 1:
        CRB_Gaussian_Natural=m/(scale**(1/2))
        CRB_Gaussian_Natural=CRB_Gaussian_Natural**2
    else:
        CRB_Gaussian_Natural=(m*(m+1))**(1/2)/(scale**(1/2))
        CRB_Gaussian_Natural=CRB_Gaussian_Natural**2

    return CRB_Gaussian_Natural

#------------------------------------------------------------------------
# Euclidean CRB with Prior Inverse Wishart
#------------------------------------------------------------------------

# Real
def CRB_Covar_Gaussian_IW_Real(Sigma_0,nu,scale_uint):
    """ A function that computes the euclidean CRB for real covar matrice estimation when
        parameter follows an inverse Wishart prior
                Inputs:
                    * Sigma_0 : scale matrix of the inverse Wishart distribution
                    * nu : degrees of liberty of the inverse Wishart 
                    * scale : array of number of training data for the covar estimation
                Outputs:
                    * the CRB """
    
    p=Sigma_0.shape[0]
    scale=scale_uint.astype(float)
    N=len(scale)
    Sigma_0_inv = inv(Sigma_0)

    # Construction basis
    Omega = basis_euc_sym_mat_real(p)
    M = Omega.shape[2] 

    # Computation of Fisher matrix of the prior
    Fiw= eval_Fiw_real(p, nu, Sigma_0, Sigma_0_inv)
    
    CRB_Gaussian_IW=np.zeros(N)
    for n in range(N):
        F=np.zeros((M,M))
        for i in range(M):
            for j in range(M):
                temp=(scale[n]/2)*((nu**2/(nu-p-1)**2)*np.matrix.trace(Sigma_0_inv@Omega[:,:,i]@Sigma_0_inv@Omega[:,:,j]) \
                +(2*nu/(nu-p-1)**2)*np.matrix.trace(Sigma_0_inv@Omega[:,:,j])*np.matrix.trace(Sigma_0_inv@Omega[:,:,i]))+Fiw[i,j] # 2 au numerateur a verif change rien bizzarre
                F[i,j]=np.real(temp)
        CRB_Gaussian_IW[n]=np.matrix.trace(inv(F)) 
    return(CRB_Gaussian_IW)

# Complex
def CRB_Covar_Gaussian_IW_Complex(Sigma_0,nu,scale_uint):
    """ A function that computes the euclidean CRB for complex covar matrice estimation when
        parameter follows an inverse Wishart prior
                Inputs:
                    * Sigma_0 : scale matrix of the inverse Wishart distribution
                    * nu : degrees of liberty of the inverse Wishart 
                    * scale : array of number of training data for the covar estimation
                Outputs:
                    * the CRB """
    
    p=Sigma_0.shape[0]
    scale=scale_uint.astype(float)
    N=len(scale)
    Sigma_0_inv = inv(Sigma_0)

    # Construction basis
    Omega = basis_euc_sym_mat_complex(p)
    M = Omega.shape[2] 
    
    # Computation of Fisher matrix of the prior
    Fiw = eval_Fiw_complex(p, nu, Sigma_0, Sigma_0_inv)
    
    CRB_Gaussian_IW=np.zeros(N)
    CRB_Gaussian_IW_asymptotic=np.zeros(N)
    FSigma = np.zeros((M,M))
    F = np.zeros((M,M))
    for n in range(N):
        F=np.zeros((M,M))
        for i in range(M):
            for j in range(M):
                FSigma[i][j]=np.real((scale[n]*(nu**2)/(nu-p)**2)*np.matrix.trace(Sigma_0_inv@Omega[:,:,i]@Sigma_0_inv@Omega[:,:,j]) \
                +(nu*scale[n]/(nu-p)**2)*np.matrix.trace(Sigma_0_inv@Omega[:,:,j])*np.matrix.trace(Sigma_0_inv@Omega[:,:,i]))
                F[i][j]=FSigma[i][j]+Fiw[i,j]
                
        CRB_Gaussian_IW[n]=np.matrix.trace(inv(F))
        CRB_Gaussian_IW_asymptotic[n]=np.matrix.trace(inv(FSigma))
    return(CRB_Gaussian_IW,CRB_Gaussian_IW_asymptotic)

#------------------------------------------------------------------------
# Natural CRB with Prior Inverse Wishart
#------------------------------------------------------------------------

# Real 
def ICRB_Covar_Gaussian_IW_Real(Sigma_0,nu,scale_uint):
    """ A function that computes the intrinsic CRB for real covar matrice estimation when
        parameter follows an inverse Wishart prior
                Inputs:
                    * Sigma_0 : scale matrix of the inverse Wishart distribution
                    * nu : degrees of liberty of the inverse Wishart 
                    * scale : array of number of training data for the covar estimation
                Outputs:
                    * the CRB """
    
    p=Sigma_0.shape[0]
    N=len(scale_uint)
    scale=scale_uint.astype(float)
    ICRB_Gaussian_IW=np.zeros(N)

    Omega = basis_euc_sym_mat_real(p)
    M = Omega.shape[2]
    T1eval = formula_T1_real(p, nu, Omega)
    T2eval = formula_T2_real(p, nu, M)
    T4eval = formula_T4_real(p, nu, M)
    
    # Construction de la FIM F^{AI}_{prior}
    Fiw = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            Fiw[i,j] = 0.25*(((nu+p+1)**2)*T1eval[i]*T1eval[j]\
                +((nu-p-1)**2)*T4eval[i,j]\
                -(nu+p+1)*(nu-p-1)*T1eval[i]*T2eval[j]\
                -(nu+p+1)*(nu-p-1)*T1eval[j]*T2eval[i])
    
    for n in range(N):
        F=np.zeros((M,M))
        F=(0.5)*scale[n]*np.eye(M)+Fiw 
        ICRB_Gaussian_IW[n]=np.matrix.trace(inv(F)) 
        
    return(ICRB_Gaussian_IW)

# Complex
def ICRB_Covar_Gaussian_IW_Complex(Sigma_0,nu,scale_uint):
    """ A function that computes the intrinsic CRB for complex covar matrice estimation when
        parameter follows an inverse Wishart prior
                Inputs:
                    * Sigma_0 : scale matrix of the inverse Wishart distribution
                    * nu : degrees of liberty of the inverse Wishart 
                    * scale : array of number of training data for the covar estimation
                Outputs:
                    * the CRB """
    
    p=Sigma_0.shape[0]
    N=len(scale_uint)
    scale=scale_uint.astype(float)
    ICRB_Gaussian_IW=np.zeros(N)
    ICRB_Gaussian_IW_asymptotic=np.zeros(N)


    Omega = basis_euc_sym_mat_complex(p)
    M = Omega.shape[2]
    T1eval = formula_T1_complex(p, nu, Omega)
    T2eval = formula_T2_complex(p, nu, M)
    T4eval = formula_T4_complex(p, nu, M)
    
    # Construction de la FIM F^{AI}_{prior}
    Fiw = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            Fiw[i,j] = ((nu+p)**2)*T1eval[i]*T1eval[j]\
                +((nu-p)**2)*T4eval[i,j]\
                -(nu+p)*(nu-p)*T1eval[i]*T2eval[j]\
                -(nu+p)*(nu-p)*T1eval[j]*T2eval[i]
    
    for n in range(N):
        F=np.zeros((M,M))
        F=scale[n]*np.eye(M)+Fiw 
        ICRB_Gaussian_IW[n]=np.matrix.trace(inv(F)) 
        ICRB_Gaussian_IW_asymptotic[n]=np.matrix.trace(inv(scale[n]*np.eye(M))) 
        
    return(ICRB_Gaussian_IW,ICRB_Gaussian_IW_asymptotic)

#---------------------------------------------------------------------------------
#  Euclidean F_IW
#---------------------------------------------------------------------------------

# real
def eval_Fiw_real(p, nu, Sigma_0, iSigma_0):
    """ A function that computes the Fprior for real covar matrice estimation when
        parameter follows an inverse Wishart prior
                Inputs:
                    * Sigma_0 : scale matrix of the inverse Wishart distribution
                    * iSigma_0 : inverse of Sigma_0
                    * nu : degrees of liberty of the inverse Wishart 
                Outputs:
                    * the Fprior """
    # Inputs:
    # p : data size
    # Sigma_0 : scale matrix of the inverse Wishart distribution
    # iSigma_0 : inverse of Sigma_0
    # nu : degrees of freedom of the inverse Wishart distribution 
    
    # Construction basis
    Omega = basis_euc_sym_mat_real(p)
    M = Omega.shape[2] 
    
    Fiw = np.zeros((M,M))
    for i in range(M):
        for j in range(M): 
            E_Tasbs, E_TasTbs = Expectation_TraceWishart_Ordre2_real(nu,Omega[:,:,i],Omega[:,:,j],iSigma_0)
            E_Tasbs = (nu**2/(nu-p-1)**2)*E_Tasbs
            E_TasTbs = (nu**2/(nu-p-1)**2)*E_TasTbs
            E_TasbsTcs = Expectation_TraceWishart_Ordre3_real(nu,Omega[:,:,i],Sigma_0,Omega[:,:,j],iSigma_0)
            E_TasbsTcs = (nu**3/(nu-p-1)**3)*E_TasbsTcs
            E_TasbsTcsds = Expectation_TraceWishart_Ordre4_real(nu,Omega[:,:,i],Sigma_0,Omega[:,:,j],Sigma_0,iSigma_0)
            E_TasbsTcsds = (nu**4/(nu-p-1)**4)*E_TasbsTcsds
            Fiw[i,j] = 0.25*((nu+p+1)**2)*E_TasTbs + ((nu-p-1)**2)*E_TasbsTcsds - 2*(nu-p-1)*(nu+p+1)*E_TasbsTcs
    
    return Fiw

# complex
def eval_Fiw_complex(p, nu, Sigma_0, iSigma_0):
    """ A function that computes the Fprior for complex covar matrice estimation when
        parameter follows an inverse Wishart prior
                Inputs:
                    * Sigma_0 : scale matrix of the inverse Wishart distribution
                    * iSigma_0 : inverse of Sigma_0
                    * nu : degrees of liberty of the inverse Wishart 
                Outputs:
                    * the Fprior """

    # Construction basis
    Omega = basis_euc_sym_mat_complex(p)
    M = Omega.shape[2] 
    
    Fiw = np.zeros((M,M))
    if 1:
        for i in range(M):
            for j in range(M): 
                E_Tasbs, E_TasTbs = Expectation_TraceWishart_Ordre2_complex(nu,Omega[:,:,i],Omega[:,:,j],iSigma_0)
                E_Tasbs = (nu**2/(nu-p)**2)*E_Tasbs
                E_TasTbs = (nu**2/(nu-p)**2)*E_TasTbs
                E_TasbsTcs = Expectation_TraceWishart_Ordre3_complex(nu,Omega[:,:,i],Sigma_0,Omega[:,:,j],iSigma_0)
                E_TasbsTcs = (nu**3/(nu-p)**3)*E_TasbsTcs
                E_TasbsTcsds = Expectation_TraceWishart_Ordre4_complex(nu,Omega[:,:,i],Sigma_0,Omega[:,:,j],Sigma_0,iSigma_0)
                E_TasbsTcsds = (nu**4/(nu-p)**4)*E_TasbsTcsds
                Fiw[i,j] = ((nu+p)**2)*E_TasTbs + ((nu-p)**2)*E_TasbsTcsds - 2*(nu-p)*(nu+p)*E_TasbsTcs
    else:
        alpha = (3*nu**2 - p*nu)/(nu-p)**2 
        beta = (nu**3 + p*nu**2 +2*nu)/(nu-p)**2
        
        for i in range(M):
            for j in range(M): 
                A = np.matrix.trace(iSigma_0@Omega[:,:,j])*np.matrix.trace(iSigma_0@Omega[:,:,i])
                B = np.matrix.trace(iSigma_0@Omega[:,:,i]@iSigma_0@Omega[:,:,j])
                Fiw[i,j] = alpha*A + beta*B
                
    return Fiw

#---------------------------------------------------------------------------------
# Expectation of trace of wishart (appendix A) for CRB derivations
#---------------------------------------------------------------------------------

# real
def Expectation_TraceWishart_Ordre2_real(K,A,B,Sigma):
    """ A function that computes expectation of Tr(A@S@B@S) et Tr(A@S)Tr(B@S) when all matrices are real
                Inputs:
                    * A,B: deterministic matrices
                    * Sigma : scale matrix of the inverse Wishart distribution for S
                    * K : degrees of freedom of the inverse Wishart 
                Outputs:
                    * both expectations """
    
    E_Tasbs = np.matrix.trace(A@Sigma@B@Sigma) + (2.0/K)*(np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma))
    E_TasTbs = np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma) + (2.0/K)*(np.matrix.trace(A@Sigma@B@Sigma))
    
    return E_Tasbs, E_TasTbs

def Expectation_TraceWishart_Ordre3_real(K,A,B,C,Sigma):
    """ A function that computes expectation of Tr(A@S@B@S)Tr(C@S) when all matrices are real
                Inputs:
                    * A,B,C: deterministic matrices
                    * Sigma : scale matrix of the inverse Wishart distribution for S
                    * K : degrees of freedom of the inverse Wishart 
                Outputs:
                    * the expectation """
    
    E_TasbsTcs = np.matrix.trace(A@Sigma@B@Sigma)*np.matrix.trace(C@Sigma) \
        + (2.0/K)*(np.matrix.trace(A@Sigma@B@Sigma@C@Sigma) \
          +      np.matrix.trace(A@Sigma@C@Sigma@B@Sigma) \
        + np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(C@Sigma)) \
        + (4.0/K**2)*(np.matrix.trace(A@Sigma@C@Sigma)*np.matrix.trace(B@Sigma) \
                 +  np.matrix.trace(B@Sigma@C@Sigma)*np.matrix.trace(A@Sigma))
            
    return E_TasbsTcs

def Expectation_TraceWishart_Ordre4_real(K,A,B,C,D,Sigma):
    """ A function that computes expectation of Tr(A@S@B@S)Tr(C@S@D@S) when all matrices are real
                Inputs:
                    * A,B,C,D: deterministic matrices
                    * Sigma : scale matrix of the inverse Wishart distribution for S
                    * K : degrees of freedom of the inverse Wishart 
                Outputs:
                    * the expectation """
                 
    E_TasbsTcsds = np.matrix.trace(A@Sigma@B@Sigma)*np.matrix.trace(C@Sigma@D@Sigma) \
        + (2.0/K)*(np.matrix.trace(A@Sigma@B@Sigma@C@Sigma@D@Sigma) \
        +        np.matrix.trace(A@Sigma@C@Sigma@D@Sigma@B@Sigma) \
        +        np.matrix.trace(A@Sigma@B@Sigma@D@Sigma@C@Sigma) \
        +        np.matrix.trace(A@Sigma@D@Sigma@C@Sigma@B@Sigma) \
        +        np.matrix.trace(A@Sigma@B@Sigma)*np.matrix.trace(C@Sigma)*np.matrix.trace(D@Sigma) \
        +        np.matrix.trace(C@Sigma@D@Sigma)*np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma)) \
        + (4.0/K**2)*(np.matrix.trace(A@Sigma@D@Sigma@B@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma@B@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(A@Sigma@D@Sigma)*np.matrix.trace(B@Sigma@C@Sigma) \
                +   np.matrix.trace(A@Sigma@B@Sigma@D@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(A@Sigma@B@Sigma@C@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma@B@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma)*np.matrix.trace(B@Sigma@D@Sigma) \
                +   np.matrix.trace(A@Sigma)*np.matrix.trace(D@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(B@Sigma@C@Sigma@D@Sigma)*np.matrix.trace(A@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma@D@Sigma)*np.matrix.trace(B@Sigma) \
                +   np.matrix.trace(A@Sigma@D@Sigma@C@Sigma)*np.matrix.trace(B@Sigma) \
                +   np.matrix.trace(B@Sigma@D@Sigma@C@Sigma)*np.matrix.trace(A@Sigma)) \
         + (8.0/K**3)*(np.matrix.trace(A@Sigma@D@Sigma@B@Sigma@C@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(A@Sigma@D@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(B@Sigma@C@Sigma)*np.matrix.trace(A@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(B@Sigma@D@Sigma)*np.matrix.trace(A@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma@B@Sigma@D@Sigma))
            
    return E_TasbsTcsds

# complex
def Expectation_TraceWishart_Ordre2_complex(K,A,B,Sigma):
    """ A function that computes expectation of Tr(A@S@B@S) et Tr(A@S)Tr(B@S) when all 
    matrices are complex
                Inputs:
                    * A,B: deterministic matrices
                    * Sigma : scale matrix of the inverse Wishart distribution for S
                    * K : degrees of freedom of the inverse Wishart 
                Outputs:
                    * both expectations """
    
    E_Tasbs = np.matrix.trace(A@Sigma@B@Sigma) + (1/K)*(np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma))
    E_TasTbs = np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma) + (1/K)*(np.matrix.trace(A@Sigma@B@Sigma))
    
    return E_Tasbs, E_TasTbs

def Expectation_TraceWishart_Ordre3_complex(K,A,B,C,Sigma):
    """ A function that computes expectation of Tr(A@S@B@S)Tr(C@S) when all matrices are 
        complex
                Inputs:
                    * A,B,C: deterministic matrices
                    * Sigma : scale matrix of the inverse Wishart distribution for S
                    * K : degrees of freedom of the inverse Wishart 
                Outputs:
                    * the expectation """
    
    if 1:
        E_TasbsTcs = np.matrix.trace(A@Sigma@B@Sigma)*np.matrix.trace(C@Sigma) \
        + (1/K)*(np.matrix.trace(A@Sigma@B@Sigma@C@Sigma) \
          +      np.matrix.trace(A@Sigma@C@Sigma@B@Sigma) \
        + np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(C@Sigma)) \
        + (1/K**2)*(np.matrix.trace(A@Sigma@C@Sigma)*np.matrix.trace(B@Sigma) \
                 +  np.matrix.trace(B@Sigma@C@Sigma)*np.matrix.trace(A@Sigma))
    else:
        AA = np.matrix.trace(A@Sigma)*np.matrix.trace(C@Sigma)
        BB = np.matrix.trace(A@Sigma@C@Sigma)
        p = Sigma.shape[0] 
        E_TasbsTcs = AA+(1/K)*(2*BB+p*AA)+(1/K**2)*(p*BB+AA)
                
            
    return E_TasbsTcs

def Expectation_TraceWishart_Ordre4_complex(K,A,B,C,D,Sigma):
    """ A function that computes expectation of Tr(A@S@B@S)Tr(C@S@D@S) when all matrices 
        are complex
                Inputs:
                    * A,B,C,D: deterministic matrices
                    * Sigma : scale matrix of the inverse Wishart distribution for S
                    * K : degrees of freedom of the inverse Wishart 
                Outputs:
                    * the expectation """
                 
    if 1:
        E_TasbsTcsds = np.matrix.trace(A@Sigma@B@Sigma)*np.matrix.trace(C@Sigma@D@Sigma) \
        + (1/K)*(np.matrix.trace(A@Sigma@B@Sigma@C@Sigma@D@Sigma) \
        +        np.matrix.trace(A@Sigma@C@Sigma@D@Sigma@B@Sigma) \
        +        np.matrix.trace(A@Sigma@B@Sigma@D@Sigma@C@Sigma) \
        +        np.matrix.trace(A@Sigma@D@Sigma@C@Sigma@B@Sigma) \
        +        np.matrix.trace(A@Sigma@B@Sigma)*np.matrix.trace(C@Sigma)*np.matrix.trace(D@Sigma) \
        +        np.matrix.trace(C@Sigma@D@Sigma)*np.matrix.trace(A@Sigma)*np.matrix.trace(B@Sigma)) \
        + (1/K**2)*(np.matrix.trace(A@Sigma@D@Sigma@B@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma@B@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(A@Sigma@D@Sigma)*np.matrix.trace(B@Sigma@C@Sigma) \
                +   np.matrix.trace(A@Sigma@B@Sigma@D@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(A@Sigma@B@Sigma@C@Sigma)*np.matrix.trace(D@Sigma) \
                
                +   np.matrix.trace(A@Sigma@C@Sigma)*np.matrix.trace(B@Sigma@D@Sigma) \
                +   np.matrix.trace(A@Sigma)*np.matrix.trace(D@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(B@Sigma@C@Sigma@D@Sigma)*np.matrix.trace(A@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma@D@Sigma)*np.matrix.trace(B@Sigma) \
                +   np.matrix.trace(A@Sigma@D@Sigma@C@Sigma)*np.matrix.trace(B@Sigma) \
                +   np.matrix.trace(B@Sigma@D@Sigma@C@Sigma)*np.matrix.trace(A@Sigma)) \
         + (1/K**3)*(np.matrix.trace(A@Sigma@D@Sigma@B@Sigma@C@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(A@Sigma@D@Sigma)*np.matrix.trace(B@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(B@Sigma@C@Sigma)*np.matrix.trace(A@Sigma)*np.matrix.trace(D@Sigma) \
                +   np.matrix.trace(B@Sigma@D@Sigma)*np.matrix.trace(A@Sigma)*np.matrix.trace(C@Sigma) \
                +   np.matrix.trace(A@Sigma@C@Sigma@B@Sigma@D@Sigma))
    else:
        AA = np.matrix.trace(A@Sigma)*np.matrix.trace(C@Sigma)
        BB = np.matrix.trace(A@Sigma@C@Sigma)
        p = Sigma.shape[0] 
        E_TasbsTcsds = AA+(1/K)*(4*BB+2*p*AA)+(1/K**2)*(5*p*BB+5*AA+AA*p**2)+(1/K**3)*(3*p*AA+2*BB+BB*p**2)
            
    return E_TasbsTcsds

#-------------------------------------------------------------------------------
# Computation T_1, T_2, T_3, T_4 for ICRB derivation
#-------------------------------------------------------------------------------

# real
def formula_T1_real(p, nu, Omega):
    """ A function that computes T_1 from (75) and (78) when real
                Inputs:
                    * p: matrix size
                    * nu : degrees of freedom of the inverse Wishart 
                    * Omega : euclidean basis
                Outputs:
                    * T_1 """
    
    M = Omega.shape[2]
    T1=np.zeros((M,1))
    
    for i in range(M):
        T1[i] = np.matrix.trace(Omega[:,:,i])
        
    return T1

def formula_T2_real(p, nu, M):
    """ A function that computes T_2 from (75) and (78) when real
                Inputs:
                    * p: matrix size
                    * nu : degrees of freedom of the inverse Wishart 
                    * M : size of the euclidean basis
                Outputs:
                    * T_2 """

    T2=np.zeros((M,1))
    
    for i in range(M):
        if i<p:
            T2[i] = nu+p-2*(i+1)+1 
        else:
            T2[i] = 0
    
    return T2/(nu-p-1)

def formula_T4_real(p, nu, M):
    """ A function that computes T_4 from (75) and (78) when real
                Inputs:
                    * p: matrix size
                    * nu : degrees of freedom of the inverse Wishart 
                    * M : size of the euclidean basis
                Outputs:
                    * T_4 """

    T4=np.zeros((M,M))
    
    index=0
    tab_index = np.zeros(M)
    for i in range(p):
        index=index+1
    for i in range(p):
        for j in range(i+1):
            if (i!=j):
                tab_index[index] = i
                index=index+1
                
    for i in range(M):
        for j in range(M):
            if i<p and j<p:
                if i==j:
                    T4[i][i] = (nu+p-2*(i+1)+1)*(nu+p-2*(i+1)+3)
                else:
                    T4[i][j] = (nu+p-2*(i+1)+1)*(nu+p-2*(j+1)+1)
            if i>=p and j<p:
                T4[i][j] = 0
            if j>=p and i<p:
                T4[i][j] = 0
            if i>=p and j>=p:
                if i==j:
                    m=tab_index[i]
                    T4[i][i] = 2*(nu+p-2*(m+1)+1)
                else:
                    T4[i][j] = 0
    return T4/(nu-p-1)**2

# complex
def formula_T1_complex(p, nu, Omega):
    """ A function that computes T_1 from (75) and (78) when complex
                Inputs:
                    * p: matrix size
                    * nu : degrees of freedom of the inverse Wishart 
                    * Omega : euclidean basis
                Outputs:
                    * T_1 """
    
    M = Omega.shape[2]
    T1=np.zeros((M,1))
    
    for i in range(M):
        T1[i] = np.matrix.trace(Omega[:,:,i])
        
    return T1

def formula_T2_complex(p, nu, M):
    """ A function that computes T_2 from (75) and (78) when complex
                Inputs:
                    * p: matrix size
                    * nu : degrees of freedom of the inverse Wishart 
                    * M : size of the euclidean basis
                Outputs:
                    * T_2 """
    
    T2=np.zeros((M,1))
    
    for i in range(M):
        if i<p:
            T2[i] = nu+p-2*(i+1)+1
        else:
            T2[i] = 0
    
    return T2/(nu-p)

def formula_T4_complex(p, nu, M):
    """ A function that computes T_4 from (75) and (78) when complex
                Inputs:
                    * p: matrix size
                    * nu : degrees of freedom of the inverse Wishart 
                    * M : size of the euclidean basis
                Outputs:
                    * T_4 """

    T4=np.zeros((M,M))
    
    index=0
    tab_index = np.zeros(M)
    for i in range(p):
        index=index+1
    for i in range(p):
        for j in range(i+1):
            if (i!=j):
                tab_index[index] = i
                index=index+1
    for i in range(p):
        for j in range(i+1):
            if (i!=j):
                tab_index[index] = i
                index=index+1
                
    for i in range(M):
        for j in range(M):
            if i<p and j<p:
                if i==j:
                    T4[i][i] = (nu+p-2*(i+1)+1)**2+nu-(i+1)+1
                else:
                    T4[i][j] = (nu+p-2*(i+1)+1)*(nu+p-2*(j+1)+1)
            if i>=p and j<p:
                T4[i][j] = 0
            if j>=p and i<p:
                T4[i][j] = 0
            if i>=p and j>=p:
                if i==j and i<2*p*(p+1):
                    m=tab_index[i]
                    T4[i][i] = (nu+p-2*(m+1)+1)
                if i==j and i>=2*p*(p+1):
                    m=tab_index[i]
                    T4[i][i] = (nu+p-2*(m+1)+1)
                    
    return T4/(nu-p)**2