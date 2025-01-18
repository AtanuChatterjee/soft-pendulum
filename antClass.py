import numpy as np

class Ant:
    """
    This class creates an ant which can attach to a site on a node.
    Ants pull on the node, which in turn drive the rod.
    They can reorient themselves according to the local force.
    Ants have rate parameters: {Kon, Koff, Kforget, Kconvert}
    """
    
    
    
    def __init__(self, K, f0=2.8, f_ind=0.428, nestDir=None, phiMax=25):
        self.f0 = f0             # force of a single ant
        self.f_ind = f_ind       # ising parameter
        self.phi = 0  
           # orientation of ant in relation to site normal
        self.type = 'informed'   # type of ant {'informed', 'puller', 'lifter'}
        self.nestDir = nestDir
        
        self.K = K
        
        self.phiMax = phiMax * (np.pi / 180)
        
        self.node = None         # saves the node the ant is attached to
        self.site = None         # index of site; 0 for (+)  and  1 for (-)
        self.spin = None         # 0.5 for (+)  and  -0.5 for (-)
        
        return
        
    def attach(self, node, site=None):
        assert(node.get_vacancy() > 0)
        
        if self.node != None:
            self.detach()
        
        if site == None:
            mask = (node.sites == None)
            vacant_sites = np.arange(len(node.sites))[mask]  # array of open site indices
            site = np.random.choice(vacant_sites)  # chooses random open site
        
        node.sites[site] = self
        
        self.node = node
        self.site = site
        self.spin = 0.5 - site   # 0.5 for (+)  and  -0.5 for (-)
        # self.reorient()
        
        return
    
    
    def detach(self):
        
        self.node.sites[self.site] = None
        self.site = None
        self.spin = None
        
        return

    

    def convert(self):
        
        if self.type == 'puller':
            self.type = 'lifter'
            self.phi = 0
        
        elif self.type == 'lifter':
            self.type = 'puller'
        
        
        
        return
    

    
    def reorient(self):
        if self.type == 'lifter':
            return
        
        elif self.type == 'informed':
            Reference = self.nestDir
        
        elif self.type == 'puller':
            Reference = self.node.prevForce
        
        phiRef = np.arctan2(Reference[1], Reference[0]) # angle of reference
        
        phiNode = np.arctan2(self.node.y, self.node.x)  # angle of node
        phiSite = phiNode + (np.pi * self.spin)         # angle of ant site
        
        phi = phiRef - phiSite
        
        phi0 = self.phi
        
        self.phi = np.min(np.abs([phi, self.phiMax]))
        
        self.phi *= np.sign(phi)
        
        dphi = (self.phi - phi0) / 200 # The division gives gradual reorientation
        
        self.phi = phi0 + dphi

        
        return

    

    def forget(self, w):
        #TODO: remove 'w' impelemntations of pconvert
        assert self.type == 'informed'
        
        del self.nestDir
        
        self.type = 'puller'
        
        prob_pull = 1 / (1 + self.get_pConvert())
        
        r3 = np.random.uniform(0, 1)
        
        if r3 < prob_pull:
            
            return
        
        else:
            self.type = 'lifter'
            self.phi = 0
            
        return
        
    
    
    def get_pConvert(self):
        """
        Returns the ising probability for the ant to convert.
        """
        f_loc = self.node.prevForce
        phi_eff = self.node.get_theta() + self.phi + (self.spin * np.pi)
        ant_orientation = np.array([np.cos(phi_eff), np.sin(phi_eff)])
        
        arg = np.divide(
                  np.dot(f_loc,
                         ant_orientation
                         ),                  
                  self.f_ind
                  )
        
        if self.type == 'puller':
            sign = -1
        
        elif self.type == 'lifter':
            sign = +1
        
        return self.K['convert'] * np.exp(sign * arg)
        
        
        
    
    def pullForce(self):
        if self.type == 'informed': #Informed ants always pull towards the nest!
            self.reorient()
        phi_eff = self.get_theta()
        forceDirection = np.array([np.cos(phi_eff), np.sin(phi_eff)])
        pullForce = self.f0 * forceDirection
        
        if self.type == 'puller':
            n = self.node.n
            nodes = self.node.rod.nodes
            prevEdge = nodes[n].get_coords() - nodes[n-1].get_coords()
            prevEdge /= np.linalg.norm(prevEdge)
            
            tanDir = np.array([-prevEdge[1], prevEdge[0]])
            pullForce = np.dot(pullForce, tanDir) * tanDir
        
        return pullForce
    
    
    def get_coords(self):
        
        r = 0.2 # size of ant
        x0 = self.node.x
        y0 = self.node.y
        
        theta = self.get_theta()
        
        x = x0 + (r * np.cos(theta))
        y = y0 + (r * np.sin(theta))
        
        return np.array([[x0, x], [y0, y]])        

    def get_theta(self):
        
        theta0 = self.node.get_theta()
        
        theta = theta0 + self.phi + (self.spin * np.pi)
        
        return theta

    def __repr__(self):
        
        if self.node != None:
            typ = self.type[0].upper() + self.type[1:]
            out = f'{typ} ant attached to node [{self.node.n}].'
        
        else:
            typ = self.type
            out = f'Unattached {typ} ant.'
            
        return out
     