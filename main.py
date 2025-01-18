from rodClass import Rod
from utils import Simulation
import time, numpy as np

t0 = time.time()

nestAngle = np.deg2rad(0)
nestDir = [np.cos(nestAngle),np.sin(nestAngle)]

l, w, t, Y = 5, 0.086, 0.086, 5e4

nNodes = l * 4

phiMax = 25 # simulation parameter


rod = Rod(nNodes, w, t, l, Y, nestDir, phiMax)

rod.EA = 1e4
rod.EI = 5e3


K = {'on' : 0.01, 'off' : 0.005, 'forget' : 0.055, 'convert' : 0.07} # simulation parameters
# rod.K = {'on' : 0.005, 'off' : 0.0025, 'forget' : 0.015, 'convert' : 0.07}

# #add ant to node 2:
# ant = Ant()
# ant.attach(rod.nodes[2])

Sim = Simulation(rod, K, phiMax)

forceDict = Sim.Run(100, dt=5e-5, plot=False)

print(f'Runtime = {time.time() - t0} seconds.')



#TODO: Since each event effects a only a single ant, changes are inertial, and no sudden switching occurs?
#TODO: Better handling of integration (don't need to save all timepoints, just save the previous one; saving to dict every 50 steps or so)
#TODO: dampen the forces ants feel?
#TODO: somehow make it faster
#TODO: get all possible imformation from rod upon grabbing a timepoint (different forces, types of ants, etc.)


