# Class to handle ant agents on the mean-field infrastructure.
import numpy as np
from numba import njit

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
    and are in the range [-pi, pi].
    """
    def __init__(self, NV, F0, F_IND, NEST_DIR,
                 Kon, Koff, Kforget, Kconvert, Kreorient):
        self.nv = NV
        self.f0 = F0
        self.f_ind = F_IND
        self.nestDir = NEST_DIR
        self.Kon = Kon
        self.Koff = Koff
        self.Kforget = Kforget
        self.Kconvert = Kconvert
        self.Kreorient = Kreorient
        
        self.ants = np.zeros((NV, 2))
        self.angles = np.zeros((NV, 2))
        return



    # Events
    def attachAnt(self):
        """
        This function attaches an ant to a random site.
        """
        available_sites = np.argwhere(self.ants == 0)
        if available_sites.size == 0:
            return
        else:
            site = np.random.choice(available_sites)
            self.ants[site] = 1
            return
    
    def detachAnt(self):
        """
        This function detaches an ant from a random site.
        """
        occupied_sites = np.argwhere(self.ants > 1) # Any ant that is not 'informed'.
        if occupied_sites.size == 0:
            return
        else:
            site = np.random.choice(occupied_sites)
            self.ants[site] = 0
            return
        
    def convertAnt(self):
        pass

    def forgetAnt(self):
        pass

    def reorientAnt(self):
        pass




    # Get forces
    def getFi(self, q, **kwargs):
        pass

    def getFp(self, q, **kwargs):
        pass

    

    # Get rates
    def getRon(self):
        return self.Kon * np.sum(self.ants == 0) # Empty sites.
    
    def getRoff(self):
        return self.Koff * np.sum(self.ants != 0) # Occupied sites
    
    def getRforget(self):
        return self.Kforget * np.sum(self.ants == 1) # Informed ants.

    def getRreorient(self):
        return self.Kreorient * np.sum(self.ants > 1) # Any ant that is not 'informed'.

    def getRconvert(self, F):
        # Handle conversion rate with the 2D array of forces.
        pass

     
