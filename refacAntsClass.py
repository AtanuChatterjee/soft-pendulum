import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import int8, int32, float64

# Define the data types for the class
spec = [
    ('nv', int8),
    ('f0', float64),
    ('f_ind', float64),
    ('nestDir', float64[:]),
    ('Kon', float64),
    ('Koff', float64),
    ('Kforget', float64),
    ('Kconvert', float64),
    ('Kreorient', float64),
    ('ants', int8[:, :]),
    ('angles', float64[:, :]),
]

@jitclass(spec)
class Ants:
    """
    This class keeps track of attached ants.
    The variable 'ants' is structured as follows:
    ants = [[ant0a, ant0b], [ant1a, ant1b], ..., [antNVa, antNVb]]
    where each row corresponds to a node, and NV is the number of nodes.
    The first index of each row corresponds to the (+) site, and the second to the (-) site;
    for a rod aligned with the x-axis, the (+) site is to positive y-direction.
    The ants are classified as follows:
    0: No ant.
    1: Informed ant.
    2: Puller ant.
    3: Lifter ant.
    ----------------------------------------------------------------
    The variable 'angles' is structured as follows:
    angles = [[angle0a, angle0b], [angle1a, angle1b], ..., [angleNVa, angleNVb]];
    similarly to the 'ants' variable.
    The angles are the orientation of the ant in relation to the site normal,
    and are in the range [-pi, pi]. ## Actually it is [-phiMax, phiMax]. ##
    """
    def __init__(self, NV, F0, F_IND, NEST_DIR,
                 Kon, Koff, Kforget, Kconvert, Kreorient):
        self.nv = NV
        self.f0 = F0
        self.f_ind = F_IND
        self.nestDir = NEST_DIR
        # self.phiMax = phiMax # TODO: Implement this instead of having hardcoded value
        
        self.Kon = Kon
        self.Koff = Koff
        self.Kforget = Kforget
        self.Kconvert = Kconvert
        self.Kreorient = Kreorient
        
        self.ants = np.zeros((NV, 2), dtype=np.int8)
        self.angles = np.zeros((NV, 2))
        return

    def getEdges(self, q):
        return getEdges(q)
    
    def getOrthogonalEdges(self, q):
        return getOrthogonalEdges(q)
    
    def rotate(self, v, theta):
        return rotate(v, theta)

    def getAntDirections(self, q, ants, angles):
        return getAntDirections(q, ants, angles, self.nv)

    def pConvert(self, q, F):
        return pConvert(q, F, self.ants, self.angles, self.f_ind, self.Kconvert, self.nv)
    
    def attachAnt(self):
        self.ants, self.angles = attachAnt(self.ants, self.angles, self.nv)
        return
    
    def detachAnt(self):
        self.ants, self.angles = detachAnt(self.ants, self.angles, self.nv)
        return
        
    def convertAnts(self, q, F):
        self.ants = convertAnts(q, F, self.ants, self.angles, self.f_ind, self.Kconvert, self.nv)
        return
    def forgetAnt(self):
        self.ants = forgetAnt(self.ants, self.nv)
        return

    def reorientAnts(self, q, F):
        reorientAnts(self.ants, self.angles, self.nestDir, F, self.nv)
        return

    def getRon(self):
        return self.Kon * np.sum(self.ants == 0) # Empty sites
    
    def getRoff(self):
        return self.Koff * np.sum(self.ants > 1) # Puller and lifter ants.
    
    def getRforget(self):
        return self.Kforget * np.sum(self.ants == 1) # Informed ants.

    def getRreorient(self):
        return self.Kreorient * np.sum(self.ants == 2) # Puller ants.

    def getRconvert(self, q, F):
        return np.sum(self.pConvert(q, F))
    
    def getInformedForce(self, q):
        return getInformedForce(q, self.ants, self.angles, self.f0, self.nv)
    
    def getPullerForce(self, q):
        return getPullerForce(q, self.ants, self.angles, self.f0, self.nv)

@njit
def getEdges(q):
    nodes = q.reshape(-1, 2)
    e = np.zeros((nodes.shape[0], 2))
    for i in range(1, nodes.shape[0]):
        d = nodes[i] - nodes[i - 1]
        norm = np.sqrt(d[0]**2 + d[1]**2)
        e[i] = d / norm
    return e

@njit
def getOrthogonalEdges(q):
    T = getEdges(q)
    O = np.zeros_like(T)
    O[:,0] = -T[:,1]
    O[:,1] = T[:,0]
    return O

@njit
def rotate(v, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    out = np.array([c*v[0] - s*v[1],
                    s*v[0] + c*v[1]])
    return out


@njit
def getAntDirections(q, ants, angles, nv):
    antDirections = np.zeros((nv, 2, 2))
    O = getOrthogonalEdges(q)
    for i in range(len(antDirections)):
        for j in range(2):
            if ants[i, j] == 0:
                continue
            antDirections[i, j] = rotate(O[i], angles[i, j]) * (-1)**j
    return antDirections

@njit
def pConvert(q, F, ants, angles, f_ind, Kconvert, nv):
    Pr = np.zeros((nv, 2))
    p = getAntDirections(q, ants, angles, nv)
    for i in range(nv):
        for j in range(2):
            if ants[i, j] == 2 or ants[i, j] == 3:
                pi = p[i, j]
                Fi = F[i]
                arg = (pi[0]*Fi[0] + pi[1]*Fi[1]) / f_ind
                if ants[i, j] == 2:
                    Pr[i, j] = np.exp(-arg)  # Puller ants.
                elif ants[i, j] == 3:
                    Pr[i, j] = np.exp(arg)  # Lifter ants.
    return Kconvert * Pr

@njit
def attachAnt(ants, angles, nv, phiMax=52):
    a = ants.flatten()
    an = angles.flatten()
    available_sites = np.argwhere(a == 0)
    if available_sites.size == 0:
        return ants, angles
    else:
        idx = np.random.choice(len(available_sites))
        site = available_sites[idx]
        a[site] = 1
        an[site] = np.random.uniform(-phiMax, phiMax)
        return a.reshape(nv, 2), an.reshape(nv, 2)

@njit
def detachAnt(ants, angles, nv):
    a = ants.flatten()
    an = angles.flatten()
    occupied_sites = np.argwhere(a > 0)
    if occupied_sites.size == 0:
        return ants, angles
    else:
        idx = np.random.choice(len(occupied_sites))
        site = occupied_sites[idx]
        a[site] = 0
        an[site] = 0
        return a.reshape(nv, 2), an.reshape(nv, 2)

@njit
def convertAnts(q, F, ants, angles, f_ind, Kconvert, nv):
    Pr = pConvert(q, F, ants, angles, f_ind, Kconvert, nv)
    ants_flat = ants.flatten()
    valid_ants = np.where((ants_flat == 2) | (ants_flat == 3))[0]
    for i in valid_ants:
        if np.random.rand() < Pr.flatten()[i]:
            ants_flat[i] = 6 // ants_flat[i]  # 2 -> 3, 3 -> 2
    return ants_flat.reshape(nv, 2)

@njit
def forgetAnt(ants, nv):
    a = ants.flatten()
    informed = np.argwhere(a == 1)
    if informed.size == 0:
        return ants
    else:
        idx = np.random.choice(len(informed))
        site = informed[idx]
        a[site] = 2
        return a.reshape(nv, 2)
    
@njit
def reorientAnts(ants, angles, nestDir, F, nv, phiMax=52, damping=10):
    ants_flat = ants.flatten()
    angles_flat = angles.flatten()
    for i in range(nv):
        for j in range(2):
            if ants_flat[i * 2 + j] == 1:
                refDir = nestDir
            elif ants_flat[i * 2 + j] == 2:
                refDir = F[i]
            else:
                continue
            phi0 = angles_flat[i * 2 + j]  # Current angle
            theta = np.arctan2(refDir[1], refDir[0])  # Desired angle
            phi = phi0 + damping * (theta - phi0)  # Damped (weighted) angle
            if np.abs(phi) > phiMax:
                phi = np.sign(phi) * phiMax
            angles_flat[i * 2 + j] = phi  # Update angle
    return angles_flat.reshape(nv, 2)
    
@njit
def getInformedForce(q, ants, angles, f0, nv):
    p = getAntDirections(q, ants, angles, nv)
    F = np.zeros((nv, 2))
    for i in range(nv):
        for j in range(2):
            if ants[i, j] == 1:
                F[i] += f0 * p[i, j]
    return F

@njit
def getPullerForce(q, ants, angles, f0, nv):
    p = getAntDirections(q, ants, angles, nv)
    F = np.zeros((nv, 2))
    for i in range(nv):
        for j in range(2):
            if ants[i, j] == 2:
                F[i] += f0 * p[i, j]
    return F



############################################################################################################
# Other functions (such as plotting) to be implemented outside the class.

def plotAnts(q, Ants):
    q = q.reshape(-1, 2)
    antDirections = Ants.getAntDirections(q, Ants.ants, Ants.angles)
    # antDirections[:,0] = Ants.getOrthogonalEdges(q)
    # antDirections[:,1] = -Ants.getOrthogonalEdges(q)
    colordict = {0: 'none', 1: 'g', 2: 'b', 3: 'y'}
    for i in range(Ants.nv):
        for j in range(2):
            plt.quiver(q[i,0], q[i,1], antDirections[i, j, 0], antDirections[i, j, 1], color=colordict[Ants.ants[i, j]])
    return

############################################################################################################



    
