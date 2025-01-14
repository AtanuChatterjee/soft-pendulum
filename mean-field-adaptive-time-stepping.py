import numpy as np
import seaborn as sns
from numba import njit
import matplotlib.pyplot as plt

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1.5)

###############################################################################
#                        PARAMETERS
###############################################################################
L       = 5.0                   # total length
NV      = int(L)+1              # number of nodes
DT_INIT = 1e-4                  # initial time step
P       = 3.0                   # puller strength
G       = 0.48                  # informed puller strength
GAMMA   = 2.5                   # damping
F0      = 2.8                   # single ant force
KC      = 0.7                   # conversion rate
F_IND   = 0.428                 # temperature
DP_INIT = 0.1                   # initial "difference in pullers"
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
def createRod(nv, length, Dp_init):
    """
    Create 'nodes' = (nv,3): [x, y, Dp].
    Then compute reference segment lengths 'dL'.
    """
    nodes = np.zeros((nv, 3))
    nodes[:, 0] = np.linspace(0, length, nv)
    nodes[1:, 2] = Dp_init
    diffs = nodes[1:, :2] - nodes[:-1, :2]  # shape (nv-1, 2)
    dL = np.sqrt(np.sum(diffs**2, axis=1))  # shape (nv-1,)
    return nodes, dL

@njit
def getStateVectors(nodes):
    """
    Convert node positions (x,y) into q (2*nv,)
    and node Dp into a separate array (nv,).
    """
    nv = nodes.shape[0]
    q0 = np.zeros(2*nv)
    for i in range(nv):
        q0[2*i  ] = nodes[i,0]  # x
        q0[2*i+1] = nodes[i,1]  # y
    Dp0 = np.zeros(nv)
    for i in range(1, nv):
        Dp0[i] = nodes[i,2]
    return q0, Dp0

###############################################################################
#                           FORCE ROUTINES
###############################################################################
@njit
def getFg(q, nv, f0, g, nx, ny):
    """
    Informed puller force: f0*g*(nx, ny) on each node
    """
    Fg = np.zeros_like(q)
    for k in range(nv - 1):
        idx = 2*k
        Fg[idx  ] += f0*g*nx
        Fg[idx+1] += f0*g*ny
    return Fg


@njit
def getFp(FOld, q, Dp_old, nv, dt, p, kc, F_ind, f0):
    """
      Dp_new[k] = dt*(p*kc*sinh(...) - 2*kc*Dp_old[k]*cosh(...)) + Dp_old[k]
      Fp[2*k..2*k+2] += f0*Dp_new[k]*n
    """
    Fp = np.zeros_like(q)
    Dp_new = np.copy(Dp_old)

    for k in range(nv - 1):
        Fx = FOld[2*k]
        Fy = FOld[2*k+1]

        # normal direction from q
        xk = q[2*k]
        yk = q[2*k+1]
        nx = -yk
        ny =  xk
        length_n = np.sqrt(nx*nx + ny*ny) + 1e-34
        nx /= length_n
        ny /= length_n

        dot_val = Fx*nx + Fy*ny
        Dp_new[k] = dt*( p*kc*np.sinh(dot_val/F_ind)
                         - 2.0*kc*Dp_old[k]*np.cosh(dot_val/F_ind)
                       ) + Dp_old[k]

        Fp[2*k  ] += f0*Dp_new[k]*nx
        Fp[2*k+1] += f0*Dp_new[k]*ny

    return Fp, Dp_new


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
def solveStep(q, q0, qDot, Dp0, dL,
              nv, dt, gamma, p, kc, F_ind, f0, g, EI, EA,
              nx, ny):
    """
      F = qDot
      Fp, Dp = getFp(F,...)
      Ftot = Fp + Fs + Fb + Fg
      qDot_new = Ftot / gamma
      qNew = q0 + qDot_new*dt
      hinge first node
    """
    Fb_ = getFb(q, dL, nv, EI)
    Fs_ = getFs(q, dL, nv, EA)
    Fg_ = getFg(q, nv, f0, g, nx, ny)
    F   = qDot
    Fp_, Dp_new = getFp(F, q, Dp0, nv, dt, p, kc, F_ind, f0)

    Ftot     = Fb_ + Fs_ + Fg_ + Fp_
    qDot_new = Ftot / gamma
    qNew     = q0 + qDot_new*dt

    # hinged boundary on the first node
    qNew[0] = q0[0]
    qNew[1] = q0[1]
    return qNew, Dp_new, qDot_new


###############################################################################
#                       ADAPTIVE STEP
###############################################################################
@njit
def doAdaptiveStep(q, q0, qDot, Dp0, dL,
                   nv, dt, gamma, p, kc, F_ind, f0, g, EI, EA,
                   nx, ny,
                   tol, min_dt, max_dt):
    """
      1) One big step => q_big
      2) Two half steps => q_half
      Compare error => adapt dt
    """
    dt_original = dt
    while True:
        q_big, Dp_big, qDot_big = solveStep(
            q, q, qDot, Dp0, dL,
            nv, dt_original, gamma, p, kc, F_ind, f0, g, EI, EA,
            nx, ny
        )

        half_dt = 0.5*dt_original
        q_mid, Dp_mid, qDot_mid = solveStep(
            q, q, qDot, Dp0, dL,
            nv, half_dt, gamma, p, kc, F_ind, f0, g, EI, EA,
            nx, ny
        )
        q_half, Dp_half, qDot_half = solveStep(
            q_mid, q_mid, qDot_mid, Dp_mid, dL,
            nv, half_dt, gamma, p, kc, F_ind, f0, g, EI, EA,
            nx, ny
        )

        # measure error in positions
        err = 0.0
        for i in range(len(q)):
            diff = q_big[i] - q_half[i]
            err += diff*diff
        err = np.sqrt(err)

        if err < tol:
            if err < 0.1*tol and dt_original < max_dt:
                dt_original = min(dt_original*1.5, max_dt)
            return q_big, Dp_big, qDot_big, dt_original
        else:
            dt_new = 0.5*dt_original
            if dt_new < min_dt:
                return q_big, Dp_big, qDot_big, dt_original
            dt_original = dt_new


###############################################################################
#                     MAIN RUN FUNCTION
###############################################################################
def run(totalTime=100.0, doPlot=True):
    nv      = NV
    length  = L
    dt      = DT_INIT
    gamma   = GAMMA
    p       = P
    kc      = KC
    F_ind   = F_IND
    f0      = F0
    g       = G
    Dp_init = DP_INIT
    nx, ny  = NEST_DIR[0], NEST_DIR[1]

    min_dt  = MIN_DT
    max_dt  = MAX_DT
    tol     = TOL

    # Create rod & initial states
    nodes, dL = createRod(nv, length, Dp_init)
    q0, Dp0_arr = getStateVectors(nodes)

    q    = q0.copy()
    Dp   = Dp0_arr.copy()
    qDot = (q - q0) / dt

    # For storing or plotting
    all_q  = []
    cTime  = 0.0
    stepCount = 0

    if doPlot:
        plt.figure(figsize=(5,5))

    # Time loop
    while cTime < totalTime:
        all_q.append(q.copy())

        q_new, Dp_new, qDot_new, used_dt = doAdaptiveStep(
            q, q, qDot, Dp, dL,
            nv, dt, gamma, p, kc, F_ind, f0, g, EI, EA,
            nx, ny,
            tol, min_dt, max_dt
        )

        q, Dp, qDot = q_new, Dp_new, qDot_new
        dt = used_dt
        cTime += dt
        stepCount += 1

        # plotting
        if doPlot and (stepCount % 1000 == 0):
            print(f"t = {cTime:.6f}, dt = {dt:e}")
            plotrod(q, cTime, length)

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
    plt.pause(0.001)


if __name__ == "__main__":
    run(totalTime=100.0, doPlot=True)
