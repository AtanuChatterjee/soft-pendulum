import numpy as np
import matplotlib.pyplot as plt
from antClass import Ant

class Simulation:
    global b, s, nodal
    def __init__(self, rod, K, phiMax):
        self.rod = rod
        self.K = K
        self.rod.K = K
        self.phiMax = phiMax
    def Event(self, kind):
        
        rod = self.rod
        
        if kind == 'attach': # Attachment event
            
            w = rod.get_omega()
            const = rod.get_ants('lifter').size + 1
            arg = w/const
            site_prob = np.array([np.exp(arg), np.exp(-arg)])
            site_prob = site_prob / np.sum(site_prob)
            site = np.random.choice([0, 1], p=site_prob) #TODO: Bias-- add p=site_prob
            
            ant = Ant(phiMax=self.phiMax, nestDir=rod.nestDir, K=self.K)
            nodes, node_prob = rod.get_attachableNodes(p=True)
            # TODO: add bias to sites with vacant chosen site?
            if len(nodes) == 0:
                return
            random_node = np.random.choice(nodes, p=node_prob) # more vacant nodes have higher chance of attachment
            ant.attach(random_node, site)
        
        elif kind == 'detach':
            ants = np.append(rod.get_ants('lifter'), rod.get_ants('puller'))
            if len(ants) == 0:
                return
            ant = np.random.choice(ants)
            ant.detach()
        
        elif kind == 'convert':
            ants = np.append(rod.get_ants('lifter'), rod.get_ants('puller'))
            if len(ants) == 0:
                return
            pConvert = [ant.get_pConvert() for ant in ants]
            pConvert = pConvert / np.sum(pConvert)
            pConvert = np.nan_to_num(pConvert, nan=0.0, posinf=0, neginf=0)
            
            ant = np.random.choice(ants, p=pConvert)
            ant.convert()
        
        elif kind == 'reorient':
            ants = np.append(rod.get_ants('informed'), rod.get_ants('puller'))
            if len(ants) == 0:
                return
            # TODO: add bias for informed?
            ant = np.random.choice(ants)
            ant.reorient()
        
        elif kind == 'forget':
            ants = rod.get_ants('informed')
            if len(ants) == 0:
                return
            # TODO: add bias to informed ants on the non pulling site?
            ant = np.random.choice(ants)
            ant.forget(rod.get_omega())
        
        return
            
            
    
    def Gillespie(self, nEvents=1):
        
        rod = self.rod
        
        Rattach = rod.get_Rattach()
        Rdetach = rod.get_Rdetach()
        Rconvert = rod.get_Rconvert()
        Rorient = rod.get_Rorient()
        Rforget = rod.get_Rforget()
        
        R = np.array([Rattach, Rdetach, Rconvert, Rorient, Rforget])
        Rtot = np.sum(R)
        R = R / Rtot
        
        events = ['attach', 'detach', 'convert', 'reorient', 'forget']
        
        for n in range(nEvents):
            r2 = np.random.uniform()
            for i in range(len(R)):
                curr_R = np.sum(R[:i+1])
                if r2 < curr_R:
                    eventKind = events[i]
                    self.Event(eventKind)
                    break
        
        r1 = np.random.uniform()
        tau = -(1/Rtot) * np.log(r1)
        
        return tau
            
    
    def AdvanceVerlet(self, curr_coords, prev_coords, forces, dt, gamma=1):
        
        qdot = (curr_coords - prev_coords) / dt
        
        Fdrag = gamma * qdot
        
        forces = forces - Fdrag
        
        new_coords = (2 * curr_coords) - (prev_coords) + (forces * np.power(dt, 2)) 
        
        return new_coords
    
    def AdvanceEuler(self, curr_coords, forces, dt, gamma=92):

        new_coords = curr_coords + ((forces / gamma)*dt)
        
        return new_coords
    
    
    def Run(self, runtime, dt=1e-4, plot=True):

        rod = self.rod
        t = 0        
        curr_coords = rod.get_nodeCoords()
        forceDict = {}
        while t < runtime: # main loop
            # Gillespie
            if t == 0:
                tau = self.Gillespie(nEvents=1)
            if t+dt > tau:
                tau = self.Gillespie(nEvents=1)
                tau = t + tau
            
            # integration
            forces, forceDict[t] = rod.get_Forces() #hella slow; around 97% of runtime

            new_coords = self.AdvanceEuler(curr_coords,
                                            forces, dt, gamma=5.5)
            
            #updating section; around 3% of runtime
            rod.update_nodeCoords(new_coords)
            rod.update_nodeForces(forces)

            curr_coords = new_coords

            t += dt
            #plotting
            if not plot:
                continue
            if int(t/dt) % 1000 == 0:
                plotRod(rod, title=f'$t = {t:.1f}$', nestDir=rod.nestDir)
                # fname = f"sim_{str(int(t*10)).zfill(5)}.png"
                # plt.savefig(r"frames/" + fname)
                plt.pause(0.005)
                plt.cla()
                continue
        
        
        
        return forceDict
    


def plotRod(rod, title='', nestDir=[0,0]):

    ant_color = {'informed': '#fc0084',
                'puller': '#01d5df',
                'lifter':'#957cfe'}
    #00bbee
    legend_elements = [plt.Line2D([0], [0], color=ant_color['informed'], lw=2.5, label='Informed'),
                        plt.Line2D([0], [0], color=ant_color['puller'], lw=2.5, label='Puller'),
                        plt.Line2D([0], [0], color=ant_color['lifter'], lw=2.5, label='Lifter')]
    

    plt.plot(rod.get_nodeCoords()[:,0], rod.get_nodeCoords()[:,1], '-o', color='gray', alpha=0.9)
    # plt.arrow(4,-4, rod.bendForce[-1][0], rod.bendForce[-1][1], color='black')
    # plt.arrow(4,-4, rod.stretchForce[-1][0], rod.stretchForce[-1][1], color='red')
    # plt.arrow(4,-4, rod.nodalForce[-1][0], rod.nodalForce[-1][1], color='blue')

    
    for ant in rod.get_ants():
        ant_coords = ant.get_coords()
        plt.plot(ant_coords[0], ant_coords[1],
                 color=ant_color[ant.type], linewidth=2, alpha=1)
    plt.title(title)
    plt.legend(handles=legend_elements, frameon=False)
    lim = rod.l*1.05
    plt.xlim([-lim,lim]), plt.ylim([-lim,lim])

    plt.arrow(0,0, nestDir[0], nestDir[1], shape='full', #plotting the nest direction
                length_includes_head=True, head_width=0.05, color='black')
