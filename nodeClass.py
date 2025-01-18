import numpy as np

class Node:
    """
        This class represents a node on a rod. Each node has two sites that can be occupied.
        A node has coordinates.
    """
    def __init__(self, rod, n, x, y=0):
        self.rod = rod 
        self.n = n  # index of node
        self.x = x  # x coordinate of the node
        self.y = y  # y coordinate of the node
        self.sites = np.array([None, None]) # [Site+, Site-] -- Defied as positive or negative angle direction.
        
        self.prevForce = np.zeros(2)
        
        return
    
    
    def get_coords(self):
        return np.array([self.x, self.y])
    
    
    def update_coords(self, coords):
        self.x = coords[0]
        self.y = coords[1]
        return
    
    def update_force(self, force):
        self.prevForce = force
    
    
    def get_theta(self):
        return np.arctan2(self.y, self.x)
    
    def get_vacancy(self):
        
        return (self.sites == None).sum()
    
    
    def get_ants(self, kind='all'):
        """
        Returns a 1D array of ants attached to the node.
        'kind' argument can be {'informed', 'puller', 'lifter'}
        """
        
        sites = self.sites
        
        ants = np.array([ant for ant in sites if not type(ant) == type(None)])
        
        if kind == 'all':
            return  ants
        
        else:
            mask = [(ant.type == kind) for ant in ants]
            return ants[mask]
        
        
    
    
    def get_informedForce(self):
        informedForce = np.zeros(2)
        
        informed = self.get_ants(kind='informed')
        for ant in informed:
            # ant.reorient()
            informedForce += ant.pullForce()
        
        return informedForce
    
    
    def get_pullerForce(self):
        #This function can be called after nodeForce was adjusted with informedForce...
        pullerForce = np.zeros(2)
        
        pullers = self.get_ants(kind='puller')
        for ant in pullers:
            pullerForce += ant.pullForce()
        
        return pullerForce
    
    # def get_antForce(self):
    #     #If you don't want extra informed adjustment...
    #     antForce = np.zeros(2)
        
    #     for ant in self.get_ants():
    #         # ant.reorient()
    #         antForce += ant.pullForce()
        
    #     return antForce


    
    
    def __repr__(self):
        return f'Node [{self.n}], with {self.get_vacancy()} vacant sites.'