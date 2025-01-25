import numpy as np
import seaborn as sns
from numba import njit
import matplotlib.pyplot as plt
from refacAntsClass import Ants, plotAnts

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1.5)

###############################################################################
#                        PARAMETERS
###############################################################################
L       = 5.0                   # total length
NV      = int(L)+1              # number of nodes
DT_INIT = 1e-4                  # initial time step
GAMMA   = 2.5                   # damping
F0      = 2.8                   # single ant force
Kc      = 0.007                   # conversion rate
Kon    = 0.1                   # attachment rate
Koff   = 0.01                   # detachment rate
Kforget= 0.01                   # forgetting rate
Kreorient= 0.1                 # reorientation rate
F_IND   = 0.428                 # temperature
NEST_DIR= np.array([1.0, 0.0])  # nest direction
EI      = 1e4                   # bending rigidity
EA      = 1e4                   # stretching rigidity

# For adaptive time stepping
MIN_DT  = 1e-7                  # minimum time step
MAX_DT  = 1e-4                  # maximum time step
TOL     = 1e-5                  # tolerance for adaptive step

###############################################################################
#                          CREATE ROD & STATE VECTORS
###############################################################################
@njit
def createRod(nv, length):
    """
    Create 'nodes' = (nv, 2): [x, y].
    Then compute reference segment lengths 'dL'.
    """
    nodes = np.zeros((nv, 2))
    nodes[:, 0] = np.linspace(0, length, nv)
    dL = np.ones(nv+1) * length / (nv - 1)
    return nodes, dL

@njit
def getStateVectors(nodes):
    """
    Convert node positions (x,y) into q (2*nv,).
    """
    nv = nodes.shape[0]
    q0 = np.zeros(2*nv)
    for i in range(nv):
        q0[2*i  ] = nodes[i,0]  # x
        q0[2*i+1] = nodes[i,1]  # y
    return q0

###############################################################################
#                           FORCE ROUTINES
###############################################################################

@njit
def getFs(q, dL, nv, EA):
    """
    Axial (stretching) force using your 'gradEs' approach.
    """
    Fs = np.zeros_like(q)
    for k in range(nv - 1):
        L0 = dL[k]
        x0 = q[2*k]
        y0 = q[2*k+1]
        x1 = q[2*(k+1)]
        y1 = q[2*(k+1)+1]

        dx = x1 - x0
        dy = y1 - y0
        L  = np.sqrt(dx*dx + dy*dy)
        scalar_k = (EA / L0)*(1.0 - L/L0)
        fx = scalar_k*(dx/L)
        fy = scalar_k*(dy/L)

        Fs[2*k  ] -= fx
        Fs[2*k+1] -= fy
        Fs[2*(k+1)  ] += fx
        Fs[2*(k+1)+1] += fy

    return Fs


@njit
def getFb(q, dL, nv, EI):
    """
    Bending force summing over triplets
    """
    Fb = np.zeros_like(q)
    for k in range(nv - 2):
        x0 = q[2*k]
        y0 = q[2*k+1]
        x1 = q[2*(k+1)]
        y1 = q[2*(k+1)+1]
        x2 = q[2*(k+2)]
        y2 = q[2*(k+2)+1]

        L0 = dL[k]
        p0 = np.array([x0,y0,0.0])
        p1 = np.array([x1,y1,0.0])
        p2 = np.array([x2,y2,0.0])
        e  = p1 - p0
        f  = p2 - p1
        norm_e = np.sqrt(e.dot(e)) + 1e-34
        norm_f = np.sqrt(f.dot(f)) + 1e-34
        te = e / norm_e
        tf = f / norm_f

        dot_val   = te.dot(tf)
        cross_val = np.cross(te, tf)
        kappa = 2.0*cross_val[2] / (1.0+dot_val+1e-34)

        cross_tf_te = np.cross(tf, te)
        cross_te_tf = -cross_tf_te
        denom = (1.0+dot_val+1e-34)
        DkappaDe = (1.0/norm_e)*(-kappa*te + cross_tf_te/denom)
        DkappaDf = (1.0/norm_f)*(-kappa*te + cross_te_tf/denom)
        gradKappa = np.zeros(6)
        gradKappa[0:2] = -DkappaDe[0:2]
        gradKappa[2:4] = DkappaDe[0:2] - DkappaDf[0:2]
        gradKappa[4:6] = DkappaDf[0:2]

        tmp = gradKappa*(EI*kappa/(L0*L0))
        Fb[2*k   ] -= tmp[0]
        Fb[2*k+1 ] -= tmp[1]
        Fb[2*(k+1)   ] -= tmp[2]
        Fb[2*(k+1)+1 ] -= tmp[3]
        Fb[2*(k+2)   ] -= tmp[4]
        Fb[2*(k+2)+1 ] -= tmp[5]

    return Fb

###############################################################################
#                 SINGLE STEP (Forward Euler)
###############################################################################
@njit
def solveStep(q, F, dL, nv, dt, gamma, EI, EA,
              A):
    """
      docs
    """
    global cTime
    global eventTime

    ### Gillespie algorithm ###
    if cTime >= eventTime:
        # Exponentially distributed event rate
        Ron, Roff = A.getRon(), A.getRoff()
        Rforget, Rreorient = A.getRforget(), A.getRreorient()
        Rconvert = A.getRconvert(q, F)
        R = Ron + Roff + Rforget + Rreorient + Rconvert
        tau = np.random.exponential(1/R) # time of next event

        # Choose event
        r = np.random.rand()*R
        if r < Ron:
            A.attachAnt()
        elif r < Ron + Roff:
            print("Detaching")
            # A.detachAnt()
        elif r < Ron + Roff + Rforget:
            A.forgetAnt()
        elif r < Ron + Roff + Rforget + Rconvert:
            A.convertAnts(q, F)

    # Reorient ants
    A.reorientAnts(q, F)

    ### Forces ###
    Fb_ = getFb(q, dL, nv, EI)
    Fs_ = getFs(q, dL, nv, EA)
    Fg = A.getInformedForce(q).flatten()
    Fp = A.getPullerForce(q).flatten()
    Ftot = Fb_ + Fs_ + Fg + Fp

    qDot_new = Ftot / gamma
    qNew     = q + qDot_new*dt

    # hinged boundary on the first node
    qNew[0] = q[0]
    qNew[1] = q[1]
    return qNew, Ftot.reshape(-1, 2), tau


###############################################################################
#                     MAIN RUN FUNCTION
###############################################################################
def run(totalTime=100.0, doPlot=True):
    global cTime, eventTime
    dt      = DT_INIT
    min_dt  = MIN_DT
    max_dt  = MAX_DT
    tol     = TOL

    # Create rod & initial states
    nodes, dL = createRod(NV, L)
    q0 = getStateVectors(nodes)

    q    = q0.copy()
    F = np.zeros_like(nodes)

    # Create Ants
    Ants_ = Ants(NV=NV, F0=F0, F_IND=F_IND, NEST_DIR=NEST_DIR,
                Kon=Kon, Koff=Koff, Kforget=Kforget, Kconvert=Kc, Kreorient=Kreorient)

    # For storing or plotting
    all_q  = []
    all_t = []
    cTime  = 0.0
    eventTime = 0.0
    stepCount = 0

    if doPlot:
        plt.figure(figsize=(5,5))

    # Time loop
    while cTime < totalTime:
        all_q.append(q.copy())
        all_t.append(cTime)

        q, F, tau = solveStep(
            q, F, dL, NV, dt, GAMMA, EI, EA,
            Ants_
        )

        cTime += dt
        eventTime += tau
        stepCount += 1

        # plotting
        if doPlot and (stepCount % 1000 == 0):
            print(f"t = {cTime:.6f}, dt = {dt:e}")
            plotrod(q, cTime, L)
            plotAnts(q, Ants_)
            plt.pause(0.01)


        if cTime + dt > totalTime:
            dt = totalTime - cTime

    all_q = np.array(all_q)
    # np.savez('rod_data_adaptive.npz', q=all_q)
    print(f"Done! Steps={stepCount}, final time={cTime:.4f}")


def plotrod(q, cTime, length):
    plt.clf()
    nv = len(q)//2
    x = q[0::2]
    y = q[1::2]
    plt.plot(x[0], y[0], 'o', markersize=10, color='tab:red')
    plt.plot(x, y, '-', linewidth=2, color='black', zorder=1)
    plt.xlim([-1.2*length, 1.2*length])
    plt.ylim([-1.2*length, 1.2*length])
    plt.title(f"t={cTime:.2f}")



if __name__ == "__main__":
    Ants_ = Ants(NV=NV, F0=F0, F_IND=F_IND, NEST_DIR=NEST_DIR,
                Kon=Kon, Koff=Koff, Kforget=Kforget, Kconvert=Kc, Kreorient=Kreorient)

    q = np.random.rand(2*NV)
    F = np.random.rand(NV, 2)

    run(totalTime=100.0, doPlot=True)

























# ###############################################################################
# #                       ADAPTIVE STEP
# ###############################################################################
# @njit
# def doAdaptiveStep(q, q0, qDot, dL,
#                    nv, dt, gamma, kc, F_ind, f0, EI, EA,
#                    tol, min_dt, max_dt,
#                    Ants):
#     """
#       1) One big step => q_big
#       2) Two half steps => q_half
#       Compare error => adapt dt
#     """
#     dt_original = dt
#     while True:
#         q_big, qDot_big = solveStep(
#             q, q, qDot, dL,
#             nv, dt_original, gamma, kc, F_ind, f0, EI, EA,
#             Ants
#         )

#         half_dt = 0.5*dt_original
#         q_mid, qDot_mid = solveStep(
#             q, q, qDot, dL,
#             nv, half_dt, gamma, kc, F_ind, f0, EI, EA,
#             Ants
#         )
#         q_half, qDot_half = solveStep(
#             q_mid, q_mid, qDot_mid, dL,
#             nv, half_dt, gamma, kc, F_ind, f0, EI, EA,
#             Ants
#         )

#         # measure error in positions
#         err = 0.0
#         for i in range(len(q)):
#             diff = q_big[i] - q_half[i]
#             err += diff*diff
#         err = np.sqrt(err)

#         if err < tol:
#             if err < 0.1*tol and dt_original < max_dt:
#                 dt_original = min(dt_original*1.5, max_dt)
#             return q_big, qDot_big, dt_original
#         else:
#             dt_new = 0.5*dt_original
#             if dt_new < min_dt:
#                 return q_big, qDot_big, dt_original
#             dt_original = dt_new