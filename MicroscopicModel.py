import numpy as np
import seaborn as sns
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba import int8, float64
from numba.experimental import jitclass

sns.set_theme(context='paper', style='ticks', font_scale=1.2)

##############################################################################
#                           ANTS CLASS & HELPERS
##############################################################################
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
    ants[i, j] = 0 => no ant
    ants[i, j] = 1 => informed
    ants[i, j] = 2 => puller
    ants[i, j] = 3 => lifter

    angles[i, j] stores the (radian) angle offset from local normal
    """

    def __init__(self, NV, F0, F_IND, NEST_DIR,
                 Kon, Koff, Kforget, Kconvert, Kreorient):
        self.nv       = NV
        self.f0       = F0
        self.f_ind    = F_IND
        self.nestDir  = NEST_DIR
        self.Kon      = Kon
        self.Koff     = Koff
        self.Kforget  = Kforget
        self.Kconvert = Kconvert
        self.Kreorient= Kreorient

        self.ants     = np.zeros((NV, 2), dtype=np.int8)
        self.angles   = np.zeros((NV, 2))

    # ------------------------------------------------------------------------
    def getInformedForce(self, q):
        return getInformedForce(q, self.ants, self.angles, self.f0, self.nv)

    def getPullerForce(self, q):
        return getPullerForce(q, self.ants, self.angles, self.f0, self.nv)

    # ------------------------------------------------------------------------
    def getRon(self):
        return self.Kon * np.sum(self.ants == 0)

    def getRoff(self):
        return self.Koff * np.sum((self.ants == 2) | (self.ants == 3))

    def getRforget(self):
        return self.Kforget * np.sum(self.ants == 1)

    def getRreorient(self):
        return self.Kreorient * np.sum((self.ants == 1) | (self.ants == 2))

    def getRconvert(self, q, F):
        P = pConvert(q, F, self.ants, self.angles, self.f_ind, self.Kconvert, self.nv)
        return np.sum(P)

    # ------------------------------------------------------------------------
    def attachAnt(self):
        self.ants, self.angles = attachAnt(self.ants, self.angles, self.nv)

    def detachAnt(self):
        self.ants, self.angles = detachAnt(self.ants, self.angles, self.nv)

    def forgetAnt(self):
        self.ants, self.angles = forgetAnt(self.ants, self.angles, self.nv)

    def reorientAnts(self, q, F):
        reorientAnts(self.ants, self.angles, self.nestDir, F, self.nv)

    def convertAnts(self, q, F):
        self.ants, self.angles = convertAnts(q, F, self.ants, self.angles,
                                             self.f_ind, self.Kconvert, self.nv)


###############################################################################
#                   HELPER FUNCTIONS (ANTS)
###############################################################################
@njit
def getEdges(q):
    nv = q.size // 2
    edges = np.zeros((nv, 2))
    xy = q.reshape(nv, 2)
    for i in range(1, nv):
        dx = xy[i, 0] - xy[i - 1, 0]
        dy = xy[i, 1] - xy[i - 1, 1]
        norm_ = np.sqrt(dx * dx + dy * dy) + 1e-16
        edges[i, 0] = dx / norm_
        edges[i, 1] = dy / norm_
    return edges


@njit
def getOrthogonalEdges(q):
    E = getEdges(q)
    nv = E.shape[0]
    O = np.zeros_like(E)
    for i in range(nv):
        tx, ty = E[i, 0], E[i, 1]
        O[i, 0] = -ty
        O[i, 1] = tx
    return O


@njit
def rotate(v, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([c * v[0] - s * v[1],
                     s * v[0] + c * v[1]])


@njit
def getAntDirections(q, ants, angles, nv):
    antDirections = np.zeros((nv, 2, 2))
    O = getOrthogonalEdges(q)
    for i in range(nv):
        for j in range(2):
            if ants[i, j] == 0:
                continue
            if j == 0:
                base_dir = O[i]
            else:
                base_dir = -O[i]
            rotated = rotate(base_dir, angles[i, j])
            antDirections[i, j] = rotated
    return antDirections

@njit
def getInformedForce(q, ants, angles, f0, nv):
    dirs = getAntDirections(q, ants, angles, nv)
    F = np.zeros((nv, 2))
    for i in range(nv):
        for j in range(2):
            if ants[i,j] == 1:
                F[i] += f0 * dirs[i,j]
    return F

@njit
def getPullerForce(q, ants, angles, f0, nv):
    dirs = getAntDirections(q, ants, angles, nv)
    F = np.zeros((nv, 2))
    for i in range(nv):
        for j in range(2):
            if ants[i,j] == 2:
                F[i] += f0 * dirs[i,j]
    return F

# ----------------------------------------------------------------------
@njit
def pConvert(q, F, ants, angles, f_ind, Kconvert, nv):
    p = getAntDirections(q, ants, angles, nv)
    Pr = np.zeros((nv, 2))
    for i in range(nv):
        Fi = F[i]
        for j in range(2):
            aij = ants[i,j]
            if aij == 2:
                arg = (p[i,j,0]*Fi[0] + p[i,j,1]*Fi[1]) / f_ind
                Pr[i,j] = np.exp(-arg)
            elif aij == 3:
                arg = (p[i,j,0]*Fi[0] + p[i,j,1]*Fi[1]) / f_ind
                Pr[i,j] = np.exp(arg)
    return Kconvert * Pr

# ----------------------------------------------------------------------
@njit
def attachAnt(ants, angles, nv, phiMax=np.deg2rad(52.0)):
    a = ants.flatten()
    an = angles.flatten()
    free_sites = np.where(a == 0)[0]
    if free_sites.size == 0:
        return ants, angles
    site = np.random.choice(free_sites)
    a[site] = 1
    an[site] = 0.0
    return a.reshape(nv,2), an.reshape(nv,2)

# ----------------------------------------------------------------------
@njit
def detachAnt(ants, angles, nv):
    a = ants.flatten()
    an = angles.flatten()
    occ = np.where(a > 0)[0]
    if occ.size == 0:
        return ants, angles
    site = np.random.choice(occ)
    a[site] = 0
    an[site] = 0.0
    return a.reshape(nv,2), an.reshape(nv,2)

# ----------------------------------------------------------------------
@njit
def forgetAnt(ants, angles, nv):
    a = ants.flatten()
    an = angles.flatten()
    inf_idx = np.where(a == 1)[0]
    if inf_idx.size == 0:
        return ants, angles
    site = np.random.choice(inf_idx)
    if np.random.rand() < 0.5:
        a[site] = 2
    else:
        a[site] = 3
        an[site] = 0.0
    return a.reshape(nv,2), an.reshape(nv,2)

# ----------------------------------------------------------------------
@njit
def reorientAnts(ants, angles, nestDir, F, nv,
                 phiMax=np.deg2rad(52.0)):
    for i in range(nv):
        Fi = F[i]
        normF = np.sqrt(Fi[0] * Fi[0] + Fi[1] * Fi[1])

        for j in range(2):
            aij = ants[i, j]
            if aij == 3:
                continue
            if aij == 1:
                angle_des = np.arctan2(nestDir[1], nestDir[0])
            elif aij == 2:
                angle_des = np.arctan2(Fi[1], Fi[0])
            else:
                continue

            old_angle = angles[i, j]

            new_angle = old_angle + (angle_des - old_angle)
            if new_angle > phiMax:
                new_angle = phiMax
            elif new_angle < -phiMax:
                new_angle = -phiMax

            angles[i, j] = new_angle

# ----------------------------------------------------------------------
@njit
def convertAnts(q, F, ants, angles, f_ind, Kconvert, nv):
    P = pConvert(q, F, ants, angles, f_ind, Kconvert, nv).flatten()
    a = ants.flatten()
    an = angles.flatten()
    for idx in range(a.size):
        if a[idx] == 2 or a[idx] == 3:
            if np.random.rand() < P[idx]:
                if a[idx] == 2:
                    a[idx] = 3  # lifter
                    an[idx] = 0.0
                else:
                    a[idx] = 2  # puller
    return a.reshape(nv,2), an.reshape(nv,2)

##############################################################################
#                          ROD MECHANICS
##############################################################################
@njit
def createRod(nv, length):
    nodes = np.zeros((nv, 2))
    xs = np.linspace(0.0, length, nv)
    for i in range(nv):
        nodes[i,0] = xs[i]
        nodes[i,1] = 0.0
    dL = (length/(nv-1)) * np.ones(nv-1)
    return nodes, dL

@njit
def getStateVectors(nodes):
    nv = nodes.shape[0]
    q = np.zeros(2*nv)
    for i in range(nv):
        q[2*i  ] = nodes[i,0]
        q[2*i+1] = nodes[i,1]
    return q

@njit
def getFs(q, dL, nv, EA):
    """
    Stretching force
    """
    Fs = np.zeros_like(q)
    for k in range(nv-1):
        L0 = dL[k]
        x0, y0 = q[2*k],   q[2*k+1]
        x1, y1 = q[2*(k+1)], q[2*(k+1)+1]
        dx = x1 - x0
        dy = y1 - y0
        L = np.sqrt(dx*dx + dy*dy) + 1e-16

        fac = (EA/L0)*(1.0 - L/L0)
        fx = fac*(dx/L)
        fy = fac*(dy/L)
        Fs[2*k  ]   -= fx
        Fs[2*k+1]   -= fy
        Fs[2*(k+1)] += fx
        Fs[2*(k+1)+1] += fy
    return Fs

@njit
def getFb(q, dL, nv, EI):
    """
    Bending force
    """
    Fb = np.zeros_like(q)
    for k in range(nv-2):
        x0, y0 = q[2*k],   q[2*k+1]
        x1, y1 = q[2*(k+1)], q[2*(k+1)+1]
        x2, y2 = q[2*(k+2)], q[2*(k+2)+1]
        L0 = dL[k]

        p0 = np.array([x0, y0, 0.0])
        p1 = np.array([x1, y1, 0.0])
        p2 = np.array([x2, y2, 0.0])
        e  = p1 - p0
        f  = p2 - p1
        norm_e = np.sqrt(e.dot(e)) + 1e-16
        norm_f = np.sqrt(f.dot(f)) + 1e-16
        te = e / norm_e
        tf = f / norm_f
        dot_val   = te.dot(tf)
        cross_val = np.cross(te, tf)
        kappa = 2.0 * cross_val[2] / (1.0 + dot_val + 1e-16)

        cross_tf_te = np.cross(tf, te)
        cross_te_tf = -cross_tf_te
        denom = (1.0 + dot_val + 1e-16)
        DkappaDe = (1.0/norm_e)*(-kappa*te + cross_tf_te/denom)
        DkappaDf = (1.0/norm_f)*(-kappa*te + cross_te_tf/denom)

        gradKappa = np.zeros(6)
        gradKappa[0:2] = -DkappaDe[0:2]
        gradKappa[2:4] = DkappaDe[0:2] - DkappaDf[0:2]
        gradKappa[4:6] = DkappaDf[0:2]

        tmp = EI*kappa/(L0*L0) * gradKappa
        Fb[2*k   ] -= tmp[0]
        Fb[2*k+1 ] -= tmp[1]
        Fb[2*(k+1)   ] -= tmp[2]
        Fb[2*(k+1)+1 ] -= tmp[3]
        Fb[2*(k+2)   ] -= tmp[4]
        Fb[2*(k+2)+1 ] -= tmp[5]
    return Fb

##############################################################################
#             SINGLE STEP FOR ROD + GILLESPIE EVENT
##############################################################################
@njit
def solveStep(q, F, dL, nv, dt, gamma, EI, EA,
              antsObj,
              cTime, eventTime):

    if cTime >= eventTime:
        Ron     = antsObj.getRon()
        Roff    = antsObj.getRoff()
        Rforget = antsObj.getRforget()
        Rreorient = antsObj.getRreorient()
        Rconvert  = antsObj.getRconvert(q, F)
        Rtot = Ron + Roff + Rforget + Rreorient + Rconvert

        if Rtot > 1e-16:
            r1 = np.random.rand()
            tau = -np.log(r1)/Rtot
            eventTime = cTime + tau

            r2 = np.random.rand()

            if r2 < Ron / Rtot:
                antsObj.attachAnt()
            elif Ron / Rtot <= r2 < (Ron + Roff) / Rtot:
                antsObj.detachAnt()
            elif (Ron + Roff) / Rtot <= r2 < (Ron + Roff + Rconvert) / Rtot:
                antsObj.convertAnts(q, F)
            elif (Ron + Roff + Rconvert) / Rtot <= r2 < (Ron + Roff + Rconvert + Rreorient) / Rtot:
                antsObj.reorientAnts(q, F)
            elif Rforget != 0:
                antsObj.forgetAnt()

    Fb_ = getFb(q, dL, nv, EI)
    Fs_ = getFs(q, dL, nv, EA)
    Finf = antsObj.getInformedForce(q).flatten()
    Fpull= antsObj.getPullerForce(q).flatten()

    Ftot = Fb_ + Fs_ + Finf + Fpull
    qDot = Ftot / gamma
    qNew = q + qDot*dt
    qNew[0] = q[0]
    qNew[1] = q[1]
    return qNew, Ftot.reshape(-1,2), eventTime


##############################################################################
#                      PLOT UTILITIES
##############################################################################
from antMarker import getMarker

def plotrod(q, cTime, length):
    plt.clf()
    nv = len(q)//2
    x = q[0::2]
    y = q[1::2]
    plt.plot(x[0], y[0], 'o', markersize=10, color='gray')
    plt.plot(x, y, '-', linewidth=2, color='black', zorder=1)
    plt.xlim([-1.2*length, 1.2*length])
    plt.ylim([-1.2*length, 1.2*length])
    plt.title(f"t={cTime:.2f}")

def plotAnts(q, AntsObj):
    nv = len(q)//2
    xy = q.reshape(nv, 2)
    dirs = getAntDirections(q, AntsObj.ants, AntsObj.angles, nv)
    colordict = {0:'none', 1:'#fc0084', 2:'#01d5df', 3:'#957cfe'}
    for i in range(1, nv):
        for j in range(2):
            aij = AntsObj.ants[i,j]
            if aij == 0:
                continue
            sx, sy = xy[i, 0], xy[i, 1]
            ex = sx + 0.9 * dirs[i, j, 0]
            ey = sy + 0.9 * dirs[i, j, 1]
            px = np.mean([sx, ex])
            py = np.mean([sy, ey])
            angij = np.arctan2(dirs[i, j, 1], dirs[i, j, 0]) - np.pi/2
            plt.plot([px], [py],
                     color=colordict[aij],
                     linewidth=0,
                     marker=getMarker(angij),
                     markersize=7.5,
                     markevery=2,
                     alpha=0.5)
            lgnd = [Line2D([], [], color=colordict[i+1], marker=getMarker(0), markersize=12,
                          linestyle='None', label=kind) for i, kind in enumerate(['Informed', 'Puller', 'Lifter'])]
            plt.legend(handles=lgnd, loc='upper left', frameon=False, ncols=len(lgnd))

##############################################################################
#                       MAIN RUN FUNCTION
##############################################################################
def run(totalTime, doPlot=False, saveData=True):
    L = 15.0
    NV = int(L) + 1
    EI = 1e4
    EA = 1e4
    gamma = 2.5
    dt = 1e-4

    nodes, dL = createRod(NV, L)
    q0 = getStateVectors(nodes)

    F0 = 2.8
    F_IND = 10.0
    nestDir = np.array([1.0, 0.0])
    Kon = 0.0215
    Koff = 0.015
    Kforget = 0.09
    Kconvert = 1.0
    Kreorient = 0.7

    antsObj = Ants(NV=NV, F0=F0, F_IND=F_IND, NEST_DIR=nestDir,
                   Kon=Kon, Koff=Koff, Kforget=Kforget,
                   Kconvert=Kconvert, Kreorient=Kreorient)

    # Time stepping
    cTime = 0.0
    eventTime = 0.0
    q = q0.copy()
    F = np.zeros((NV, 2))

    time_list = []
    q_list = []
    ants_list = []
    angles_list = []

    stepCount = 0

    if doPlot:
        plt.figure(figsize=(6, 6))

    while cTime < totalTime:
        if stepCount % 10000 == 0:
            time_list.append(cTime)
            q_list.append(q.copy())
            ants_list.append(antsObj.ants.copy())
            angles_list.append(antsObj.angles.copy())

        q, F, eventTime = solveStep(q, F, dL, NV, dt, gamma, EI, EA,
                                    antsObj,
                                    cTime, eventTime)
        cTime += dt
        stepCount += 1

        if doPlot and (stepCount % 10000 == 0):
            print(f"time={cTime:.4f}")
            plotrod(q, cTime, L)
            plotAnts(q, antsObj)
            plt.pause(0.01)

    time_list.append(cTime)
    q_list.append(q.copy())
    ants_list.append(antsObj.ants.copy())
    angles_list.append(antsObj.angles.copy())

    if saveData:
        time_array = np.array(time_list)
        q_array = np.array(q_list)
        ants_array = np.array(ants_list)
        angles_array = np.array(angles_list)

        np.savez("soft_pendulum_microscopic.npz",
                 time=time_array,
                 q=q_array,
                 ants=ants_array,
                 angles=angles_array)
        print("Data saved!")

if __name__ == "__main__":
    run(totalTime=400, doPlot=True)
