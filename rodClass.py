import numpy as np
import pandas as pd
from nodeClass import Node
from antClass import Ant
from mechForces import bendForces, stretchForces

class Rod:
    """
        This class creates the rod with the given parameters, and lines it with nodes.
    """

    def __init__(self, nNodes, w, t, l, Y, nestDir, phiMax):
        self.nNodes = nNodes  # Number of nodes in the rod
        self.w = w  # Width of the rod
        self.t = t  # Thickness of the rod
        self.l = l  # Length of the rod
        self.Y = Y  # Young's modulus of the rod
        self.EA = Y * w * t  # Axial stiffness of the rod
        self.EI = Y * (w * t ** 3) / 12  # Flexural stiffness of the rod
        self.nestDir = nestDir  # Direction of the nest
        self.phiMax = phiMax * (np.pi / 180) # Maximal ant orientation relative to site normal (degrees)
        self.timeDict = {'bend' : [], 'stretch' : [], 'nodal' : []}
        from time import time
        self.getTime = time
        
        self.K = {'on' : 0.12, 'off' : 0.015, 'forget' : 0.035, 'convert' : 0.7}
        

        
        self.nodes = np.array([Node(self, n, x) 
                              for n, x in zip(range(nNodes+1),
                                              np.linspace(0, l, num=nNodes+1))]
                              ) #Creates origin Node + nNodes
        # self.refLen = np.linalg.norm(np.diff(self.get_nodeCoords(), axis=0), axis=1).mean()  # Reference length of each element
        self.refLen = l / nNodes # Reference length of each element
        self.prevForce = np.zeros((nNodes + 1, 2)) # array of initial\previous forces along each node.
        
        self.theta = self.get_theta(mean=False)
        
        self.bendForce = []
        self.stretchForce = []
        self.nodalForce = []
        
        return
    
        
    def get_nodeCoords(self, as_DataFrame = False):
        """
        Returns pandas DataFrame with node coordinates.
        """
        
        arr = np.array([node.get_coords() for node in self.nodes])
        
        if as_DataFrame:
            return pd.DataFrame(data=arr, columns=['x','y'])
        else:
            return arr
    
    def update_nodeCoords(self, coords):
        nodes = self.nodes
        for i in range(len(nodes)):
            node = nodes[i]
            node.update_coords(coords[i])
        return
    
    def update_nodeForces(self, forces):
        nodes = self.nodes
        for i in range(len(nodes)):
            node = nodes[i]
            node.update_force(forces[i])
        return
        
    
    
    def get_theta(self, mean=True):
        """
        Returns the mean rod angle in relation to the x-axis.
        If the angle for each node is desired, use mean=False.
        """
        arr = np.array([node.get_theta() for node in self.nodes[1:]])
        if mean:
            return arr.mean()
        else:
            return arr
        
        
    def get_omega(self, mean=True):
        if mean:
            w = self.get_theta() - self.theta.mean() # Current angle minus previous angle
        else:
            w = self.get_theta(mean=False) - self.theta
        return w
    
    
    
    def get_attachableNodes(self, p=False):
        
        attachableNodes = [node for node in self.nodes[1:] if node.get_vacancy() > 0]
        
        if not p:
            return attachableNodes
        else:
            node_prob = np.array([node.get_vacancy() for node in attachableNodes])
            node_prob = node_prob / np.sum(node_prob)
            return attachableNodes, node_prob
            
    
    def get_ants(self, kind='all'):
        """
        Returns an array containing all ants attached to the rod nodes.
        Resulting array of in shape (nNodes, 2); in a node by node basis.
        """
        ants = np.array([])
        for node in self.nodes:
            ants = np.append(ants, node.get_ants(kind=kind))
        
        return ants
    
    def countAnts(self):
        """
        Returns a dictionary containting the number of ants according to their type.
        """
        d = {'informed' : 0, 'puller' : 0, 'lifter' : 0}
        
        for ant in self.get_ants():
            d[ant.type] = d.get(ant.type, 0) + 1
        
        return d

    
    def get_Rattach(self):
        empty_sites = 0
        for node in self.nodes:
            empty_sites += node.get_vacancy()
        
        return empty_sites * self.K['on']
    
    def get_Rdetach(self):
        return len(self.get_ants()) * self.K['off']
    
    def get_Rforget(self):
        return self.countAnts()['informed'] * self.K['forget']
    
    def get_Rorient(self):
        d = self.countAnts()
        return (d['puller'] + d['informed']) * self.K['convert']
    
    def get_Rconvert(self):
        S = 0
        for ant in self.get_ants():
            
            s = 0
            
            phi = ant.get_theta()                              # relative angle of ant to node normal
            p = np.array( [np.cos(phi) , np.sin(phi)] )   # p is the ant orientation vector
            f_loc = self.prevForce[ant.node.n]            # f_loc is the local force at that ant's node
            
            arg =  ( np.dot(p, f_loc) ) / ( ant.f_ind )   # self-explanatory; according to previous papers
            
            if ant.type == 'puller':
                s = np.exp(-arg)
                
            elif ant.type == 'lifter':
                s = np.exp(arg)
            
            S += s
        
        return self.K['convert'] * S


    
    
    def get_bendForces(self):
        Fb = bendForces(self.get_nodeCoords(), self.EI, self.refLen)
        return Fb
        
    
    def get_stretchForces(self): 
        Fs = stretchForces(self.get_nodeCoords(), self.EA, self.refLen)
        return Fs
        
    
    def get_nodalForces(self):
        
        nodalForces = np.zeros((len(self.nodes), 2))
        
        for node in self.nodes:
            informedForce = node.get_informedForce()
            pullerForce = node.get_pullerForce()
            
            n = node.n
            
            nodalForces[n] = informedForce + pullerForce

        return nodalForces
    
    
    def get_Forces(self):
        t0 = self.getTime()
        Fb = self.get_bendForces()          * 1
        t1 = self.getTime()
        Fs = self.get_stretchForces()       * 1
        t2 = self.getTime()
        Fnodal = self.get_nodalForces()     * 1
        t3 = self.getTime()

        self.timeDict['bend'].append(t1 - t0) 
        self.timeDict['stretch'].append(t2 - t1)
        self.timeDict['nodal'].append(t3 - t2)

        Ftot = Fb + Fs + Fnodal
        
        if (Ftot[0] != 0).any(): # nullifies hinge node forces
            Ftot[0] = np.array([0,0])
        
        return Ftot, {'bend' : Fb, 'stretch' : Fs, 'nodal' : Fnodal}
    
    
    
    
    def __repr__(self):
        return f'Rod with {self.nNodes} Nodes, and {len(self.get_ants())} ants currently attached.'
    
    
    
    