import numpy as np
from numba import njit

#@njit
def bendForces(q, EI, refLen):
    '''
    Obtain bending force array for 2D rod
        :param q: array of position coordinates
        :param EI: flexural rigidity
        :param refLen: un-stretched length of an edge
        :return: bending force array
    '''
    q = np.hstack((q, np.zeros((q.shape[0], 1))))  # add z-coordinates to the node coordinates
    e1 = q[1:] - q[:-1]  # edge vectors e1
    e2 = q[2:] - q[1:-1]  # edge vectors e2
    
    norm_e1 = np.sqrt(np.sum(e1**2, axis=1)).reshape(-1, 1)
    norm_e2 = np.sqrt(np.sum(e2**2, axis=1)).reshape(-1, 1)
    
    t1 = e1 / norm_e1
    t2 = e2 / norm_e2
    
    d_prod = np.sum(t1 * t2, axis=1).reshape(-1, 1)
    d_prod = np.clip(d_prod, -1, 1)
    
    c_prod = np.cross(t1, t2)
    c_prod = np.clip(c_prod, -1, 1)
    
    k = 2.0 * c_prod / (1.0 + d_prod + 1e-34)
    kappa = k[:, 2:]
    
    DkappaDe1 = 1.0 / norm_e1 * (-kappa * t1 + c_prod / (1.0 + d_prod + 1e-34))
    DkappaDe2 = 1.0 / norm_e2 * (-kappa * t1 - c_prod / (1.0 + d_prod + 1e-34))
    
    DkappaDe1 = DkappaDe1[:, :2]
    DkappaDe2 = DkappaDe2[:, :2]
    
    DkappaDe1 *= kappa
    DkappaDe2 *= kappa
    
    gradKappa = np.zeros_like(q)[:, :2]
    gradKappa[2:] -= DkappaDe1
    gradKappa[1:-1] += DkappaDe1 - DkappaDe2
    gradKappa[:-2] += DkappaDe2
    
    Fb = gradKappa * EI / refLen
    return Fb

#@njit
def stretchForces(q, EA, refLen):
    '''
    Obtain stretching force array for 2D rod
        :param q: array of position coordinates
        :param EA: stretching stiffness
        :param refLen: un-stretched length of an edge
        :return: stretching force array
    '''
    
    dq = q[1:] - q[:-1]  # difference in position coordinates
    norm_dq = np.sqrt(np.sum(dq**2, axis=1))  # norm of dq
    k = EA * (1 - norm_dq / refLen).reshape(-1, 1)  # stiffness
       
    F = k * dq / refLen
    Fs = np.zeros_like(q)
    Fs[:-1] -= F
    Fs[1:] += F
    return Fs
