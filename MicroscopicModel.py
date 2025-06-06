import json
import matplotlib
import numpy as np
import os, datetime
from time import time
from tqdm import tqdm
import seaborn as sns
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba import int8, int16, float64
from numba.experimental import jitclass

np.random.seed(42)

matplotlib.use("TkAgg")
sns.set_theme(context='paper', style='ticks', font_scale=1.2)

SEG_LEN_CM = 1.0
BASE_SEG_LEN_CM = 1.0

EA_BASE = 1e4
EI_BASE = 1e4
GAMMA_BASE = 10


@njit(float64(float64))
def safe_sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


spec = [
    ('nv', int16),
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
    ('phiDamping', float64),
    ('phiMax', float64),
]


@jitclass(spec)
class Ants:
    """
    ants[i, j] = 0 => no ant
    ants[i, j] = 1 => informed
    ants[i, j] = 2 => puller
    ants[i, j] = 3 => lifter
    """

    def __init__(self, NV, F0, F_IND, NEST_DIR,
                 Kon, Koff, Kforget, Kconvert, Kreorient,
                 phiDamping, phiMax):
        self.nv = NV
        self.f0 = F0
        self.f_ind = F_IND
        self.nestDir = NEST_DIR
        self.Kon = Kon
        self.Koff = Koff
        self.Kforget = Kforget
        self.Kconvert = Kconvert
        self.Kreorient = Kreorient
        self.phiDamping = phiDamping
        self.phiMax = phiMax

        self.ants = np.zeros((NV, 2), dtype=np.int8)
        self.angles = np.zeros((NV, 2))

        # Initial conditions 
        self.ants[:, :] = 0 # no ants
        self.angles[:, :] = 0


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
        P = pConvert(q, F, self.ants, self.angles, self.f_ind, self.nv)
        return self.Kconvert * np.sum(P)

    # ------------------------------------------------------------------------
    def attachAnt(self):
        self.ants, self.angles = attachAnt(self.ants, self.angles, self.nv)

    def detachAnt(self):
        self.ants, self.angles = detachAnt(self.ants, self.angles, self.nv)

    def forgetAnt(self, q, F):
        self.ants, self.angles = forgetAnt(q, F, self.ants, self.angles,
                                           self.f_ind, self.nv, nestDir=self.nestDir,
                                           phiDamping=self.phiDamping, phiMax_deg=self.phiMax)

    def reorientAnts(self, q, F):
        reorientAnts(self.ants, self.angles, self.nestDir, self.nv,
                     q, F, self.phiDamping, self.phiMax)

    def convertAnt(self, q, F):
        self.ants, self.angles = convertAnt(q, F, self.ants, self.angles,
                                            self.f_ind, self.nv)


@njit
def getEdges(q):
    nv = q.size // 2
    xy = q.reshape(nv, 2)
    edges = np.empty((nv - 1, 2))
    for i in range(nv - 1):
        dx = xy[i + 1, 0] - xy[i, 0]
        dy = xy[i + 1, 1] - xy[i, 1]
        n = 1.0 / (np.sqrt(dx * dx + dy * dy) + 1e-16)
        edges[i, 0] = dx * n
        edges[i, 1] = dy * n
    return edges


@njit
def getOrthogonalEdges(q):
    E = getEdges(q)
    O = np.empty_like(E)
    for i in range(E.shape[0]):
        tx, ty = E[i]
        O[i, 0] = -ty
        O[i, 1] = tx
    return O


@njit
def rotate(v, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([c * v[0] - s * v[1],
                     s * v[0] + c * v[1]])


@njit(fastmath=True)
def getAntDirections(q, ants, angles, nv):
    xy = q.reshape(nv, 2)
    dirs = np.zeros((nv, 2, 2), dtype=q.dtype)

    for i in range(nv):
        if np.all(ants[i] == 0):
            continue
        n = local_normal(xy, i, nv)
        for j in (0, 1):
            if ants[i, j] == 0:
                continue
            base = n if j == 0 else -n
            vec = rotate(base, angles[i, j])
            inv = 1.0 / (np.sqrt(vec[0] ** 2 + vec[1] ** 2) + 1e-16)
            dirs[i, j, 0] = vec[0] * inv
            dirs[i, j, 1] = vec[1] * inv
    return dirs


@njit
def local_normal(xy, i, nv):
    if i == 0:
        tx, ty = xy[1] - xy[0]
    elif i == nv - 1:
        tx, ty = xy[nv - 1] - xy[nv - 2]
    else:
        tx, ty = xy[i + 1] - xy[i - 1]
    nx, ny = -ty, tx
    inv = 1.0 / (np.sqrt(nx * nx + ny * ny) + 1e-16)
    return np.array((nx * inv, ny * inv))


@njit
def getInformedForce(q, ants, angles, f0, nv):
    dirs = getAntDirections(q, ants, angles, nv)
    F = np.zeros((nv, 2), dtype=q.dtype)
    for i in range(nv):
        for j in range(2):
            if ants[i, j] == 1:
                F[i] += f0 * dirs[i, j]
    return F


@njit
def getPullerForce(q, ants, angles, f0, nv):
    dirs = getAntDirections(q, ants, angles, nv)
    F = np.zeros((nv, 2), dtype=q.dtype)
    for i in range(nv):
        for j in range(2):
            if ants[i, j] == 2:
                F[i] += f0 * dirs[i, j]
    return F


@njit
def pConvert(q, F, ants, angles, f_ind, nv):
    dirs = getAntDirections(q, ants, angles, nv)
    Pr = np.zeros((nv, 2))

    for i in range(nv):
        Fi = F[i]
        for j in range(2):
            aij = ants[i, j]
            if aij < 2:
                continue
            arg = (dirs[i, j, 0] * Fi[0] + dirs[i, j, 1] * Fi[1]) / f_ind
            p_pull = safe_sigmoid(arg)
            p_lift = 1.0 - p_pull
            Pr[i, j] = p_lift if aij == 2 else p_pull
    return Pr


@njit
def attachAnt(ants, angles, nv, phiMax=np.deg2rad(52.0)):
    free = np.argwhere(ants == 0)[2:] # Ignore origin node
    if free.shape[0] == 0:
        return ants, angles
    i, j = free[np.random.randint(free.shape[0])]
    ants[i, j] = 1
    angles[i, j] = np.random.uniform(-phiMax, phiMax)
    return ants, angles


@njit
def detachAnt(ants, angles, nv):
    a = ants.flatten()
    an = angles.flatten()
    occ = np.where((a == 2) | (a == 3))[0]
    if occ.size == 0:
        return ants, angles
    site = occ[np.random.randint(occ.size)]
    a[site] = 0
    an[site] = 0.0
    return a.reshape(nv, 2), an.reshape(nv, 2)


@njit
def forgetAnt(q, F, ants, angles, f_ind, nv,
              nestDir=np.array([1.0, 0.0]),
              phiDamping=0.95, phiMax_deg=52.0):
    a = ants.flatten()
    an = angles.flatten()
    inf = np.where(a == 1)[0]
    if inf.size == 0:
        return ants, angles
    site = inf[np.random.randint(inf.size)]

    dirs = getAntDirections(q, ants, angles, nv)
    i_vertex = site // 2
    j_side = site % 2
    pij = dirs[i_vertex, j_side]
    Fi = F[i_vertex]
    arg = (pij[0] * Fi[0] + pij[1] * Fi[1]) / f_ind

    p_pull = safe_sigmoid(-arg)
    if np.random.rand() < p_pull:
        a[site] = 2
        ants_tmp = a.reshape(nv, 2)
        angles_tmp = an.reshape(nv, 2)
        reorientAnts(ants_tmp, angles_tmp, nestDir, nv,
                     q, F, phiDamping, phiMax_deg)
        return ants_tmp, angles_tmp
    else:
        a[site] = 3
        an[site] = 0.0
    return a.reshape(nv, 2), an.reshape(nv, 2)


@njit
def signed_angle(v1, v2):
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return np.arctan2(cross, dot)


@njit
def reorientAnts(ants, angles, nestDir, nv, q, F,
                 phiDamping=0.95, phiMax_deg=52.0):
    phiMax = np.deg2rad(phiMax_deg)
    nestDir = nestDir / np.linalg.norm(nestDir)
    antDirs = getAntDirections(q, ants, angles, nv)

    for i in range(nv):
        for j in range(2):
            state = ants[i, j]
            if state == 0 or state == 3:
                continue

            base = antDirs[i, j]
            if state == 1:
                desired = nestDir
            else:
                Fi = F[i]
                Fi_ = Fi - (Fi @ base) * base
                if np.linalg.norm(Fi_) < 1e-12:
                    continue
                desired = Fi_ / np.linalg.norm(Fi_)

            phi_des = signed_angle(base, desired)
            phi_new = phiDamping * angles[i, j] + (1.0 - phiDamping) * phi_des

            # wrap & clamp
            if phi_new > np.pi:
                phi_new -= 2 * np.pi
            elif phi_new < -np.pi:
                phi_new += 2 * np.pi
            if phi_new > phiMax:
                phi_new = phiMax
            elif phi_new < -phiMax:
                phi_new = -phiMax
            angles[i, j] = phi_new


@njit
def convertAnt(q, F, ants, angles, f_ind, nv):
    P = pConvert(q, F, ants, angles, f_ind, nv).flatten()
    a = ants.flatten()
    an = angles.flatten()

    idx = np.where(a >= 2)[0]
    if idx.size == 0:
        return ants, angles
    site = idx[np.random.randint(idx.size)]
    if np.random.rand() < P[site]:
        if a[site] == 2:  # puller to lifter
            a[site] = 3
            an[site] = 0.0
        else:  # lifter to puller
            a[site] = 2
            iv, js = divmod(site, 2)
            F_loc = F[iv]
            if np.linalg.norm(F_loc) > 1e-12:
                ants_tmp = a.reshape(nv, 2)
                angles_tmp = an.reshape(nv, 2)
                dirs = getAntDirections(q, ants_tmp, angles_tmp, nv)
                base = dirs[iv, js]
                angles_tmp[iv, js] = signed_angle(base,
                                                  F_loc / np.linalg.norm(F_loc))
                return ants_tmp, angles_tmp
    return a.reshape(nv, 2), an.reshape(nv, 2)


##############################################################################
#                              ROD MECHANICS
##############################################################################
@njit
def createRod(nv, length):
    nodes = np.zeros((nv, 2))
    xs = np.linspace(0.0, length, nv)
    for i in range(nv):
        nodes[i, 0] = xs[i]
        nodes[i, 1] = 0.0
    dL = (length / (nv - 1)) * np.ones(nv - 1)
    return nodes, dL


@njit
def getStateVectors(nodes):
    nv = nodes.shape[0]
    q = np.zeros(2 * nv)
    for i in range(nv):
        q[2 * i] = nodes[i, 0]
        q[2 * i + 1] = nodes[i, 1]
    return q


@njit(fastmath=True)
def getFs(q, dL, nv, EA):
    """
    Stretching force
    """
    Fs = np.zeros_like(q)
    for k in range(nv - 1):
        L0 = dL[k]
        x0, y0 = q[2 * k], q[2 * k + 1]
        x1, y1 = q[2 * (k + 1)], q[2 * (k + 1) + 1]
        dx = x1 - x0
        dy = y1 - y0
        L = np.sqrt(dx * dx + dy * dy) + 1e-16

        fac = (EA / L0) * (1.0 - L / L0)
        fx = fac * (dx / L)
        fy = fac * (dy / L)
        Fs[2 * k] -= fx
        Fs[2 * k + 1] -= fy
        Fs[2 * (k + 1)] += fx
        Fs[2 * (k + 1) + 1] += fy
    return Fs


@njit(fastmath=True)
def getFb(q, dL, nv, EI):
    """
    Bending force
    """
    Fb = np.zeros_like(q)
    for k in range(nv - 2):
        x0, y0 = q[2 * k], q[2 * k + 1]
        x1, y1 = q[2 * (k + 1)], q[2 * (k + 1) + 1]
        x2, y2 = q[2 * (k + 2)], q[2 * (k + 2) + 1]
        L0 = dL[k]

        p0 = np.array([x0, y0, 0.0])
        p1 = np.array([x1, y1, 0.0])
        p2 = np.array([x2, y2, 0.0])
        e = p1 - p0
        f = p2 - p1
        norm_e = np.sqrt(e.dot(e)) + 1e-16
        norm_f = np.sqrt(f.dot(f)) + 1e-16
        te = e / norm_e
        tf = f / norm_f
        dot_val = te.dot(tf)
        cross_val = np.cross(te, tf)
        kappa = 2.0 * cross_val[2] / (1.0 + dot_val + 1e-16)

        cross_tf_te = np.cross(tf, te)
        cross_te_tf = -cross_tf_te
        denom = (1.0 + dot_val + 1e-16)
        DkappaDe = (1.0 / norm_e) * (-kappa * te + cross_tf_te / denom)
        DkappaDf = (1.0 / norm_f) * (-kappa * te + cross_te_tf / denom)

        gradKappa = np.zeros(6)
        gradKappa[0:2] = -DkappaDe[0:2]
        gradKappa[2:4] = DkappaDe[0:2] - DkappaDf[0:2]
        gradKappa[4:6] = DkappaDf[0:2]

        tmp = EI * kappa / (L0 * L0) * gradKappa
        Fb[2 * k] -= tmp[0]
        Fb[2 * k + 1] -= tmp[1]
        Fb[2 * (k + 1)] -= tmp[2]
        Fb[2 * (k + 1) + 1] -= tmp[3]
        Fb[2 * (k + 2)] -= tmp[4]
        Fb[2 * (k + 2) + 1] -= tmp[5]
    return Fb

from antMarker import getMarker
def plotrod(q, cTime, length):
    plt.clf()
    nv = len(q) // 2
    x = q[0::2]
    y = q[1::2]
    plt.plot(x[0], y[0], 'o', markersize=10, color='tab:red')
    plt.plot(x, y, '-', linewidth=2, color='black', zorder=1)
    plt.xlim([-1.2 * length, 1.2 * length])
    plt.ylim([-1.2 * length, 1.2 * length])
    plt.title(f"t={cTime:.2f}")


def plotAnts(q, AntsObj):
    nv = len(q) // 2
    xy = q.reshape(nv, 2)
    dirs = getAntDirections(q, AntsObj.ants, AntsObj.angles, nv)
    colordict = {0: 'none', 1: '#648FFF', 2: '#DC267F', 3: '#FE6100'}
    for i in range(nv):
        for j in range(2):
            aij = AntsObj.ants[i, j]
            if aij == 0:
                continue
            sx, sy = xy[i, 0], xy[i, 1]
            ex = sx + 0.5 * dirs[i, j, 0]
            ey = sy + 0.5 * dirs[i, j, 1]
            angij = np.arctan2((ey - sy), (ex - sx)) - np.pi/2
            plt.plot([sx, ex], [sy, ey],
                     color=colordict[aij],
                     linewidth=0,
                     marker=getMarker(angij),
                     markersize=7.5,
                     markevery=2,
                     alpha=0.5)
            lgnd = [Line2D([], [], color=colordict[i+1], marker=getMarker(0), markersize=12,
                          linestyle='None', label=kind) for i, kind in enumerate(['Informed', 'Puller', 'Lifter'])]
            plt.legend(handles=lgnd, loc='upper left', frameon=False, ncols=len(lgnd))


# ----------------------------------------------------------------------
@njit
def solveStep(q, dL, nv, dt, gamma, EI, EA,
              antsObj, cTime, eventTime):
    # deterministic forces
    Fb = getFb(q, dL, nv, EI)
    Fs = getFs(q, dL, nv, EA)
    Finf = antsObj.getInformedForce(q).flatten()
    Fpull = antsObj.getPullerForce(q).flatten()
    Fcurr = Fb + Fs + Finf + Fpull

    # Gillespie events
    if cTime >= eventTime:
        Ron = antsObj.getRon()
        Roff = antsObj.getRoff()
        Rconvert = antsObj.getRconvert(q, Fcurr.reshape(nv, 2))
        Rreorient = antsObj.getRreorient()
        Rforget = antsObj.getRforget()
        Rtot = Ron + Roff + Rforget + Rreorient + Rconvert

        if Rtot > 1e-16:
            tau = -np.log(np.random.rand()) / Rtot
            eventTime = cTime + tau
            r2 = np.random.rand()

            if r2 < Ron / Rtot:
                antsObj.attachAnt()
            elif r2 < (Ron + Roff) / Rtot:
                antsObj.detachAnt()
            elif r2 < (Ron + Roff + Rconvert) / Rtot:
                antsObj.convertAnt(q, Fcurr.reshape(nv, 2))
            elif r2 < (Ron + Roff + Rconvert + Rreorient) / Rtot:
                antsObj.reorientAnts(q, Fcurr.reshape(nv, 2))
            else:
                antsObj.forgetAnt(q, Fcurr.reshape(nv, 2))

            Finf = antsObj.getInformedForce(q).flatten()
            Fpull = antsObj.getPullerForce(q).flatten()
            Fcurr = Fb + Fs + Finf + Fpull

    # semi-implicit integration
    qDot = Fcurr / gamma
    qMid = q + 0.5 * dt * qDot

    Fb_m = getFb(qMid, dL, nv, EI)
    Fs_m = getFs(qMid, dL, nv, EA)
    Finf_m = antsObj.getInformedForce(qMid).flatten()
    Fpull_m = antsObj.getPullerForce(qMid).flatten()
    Fmid = Fb_m + Fs_m + Finf_m + Fpull_m

    qDot += (Fmid - Fcurr) / (2 * gamma)
    qNew = q + dt * qDot
    qNew[0:2] = q[0:2]  # hinge pinned

    return qNew, Fmid.reshape(nv, 2), eventTime


##############################################################################
def run(totalTime: float,
        L: float,
        params: dict,
        seg_len: float = SEG_LEN_CM,
        saveData: bool = False,
        output: str = None):
    nv = int(np.ceil(L / seg_len)) + 1
    L_sim = seg_len * (nv - 1)

    scale = seg_len / BASE_SEG_LEN_CM
    EA = EA_BASE * scale
    EI = EI_BASE * scale ** 2
    gamma = GAMMA_BASE * scale

    F0 = 2.8
    Ants_per_cm = 5.0
    F0_cluster = F0 * Ants_per_cm * seg_len

    F_IND = 4

    Kon = params.get('Kon')
    Koff = params.get('Koff')
    Kforget = params.get('Kforget')
    Kconvert = params.get('Kconvert')
    Kreorient = params.get('Kreorient')

    phiDamping, phiMax = 0.95, 52.0
    dt = 2e-4
    saveEvery = 200

    if saveData:
        print(f"Saving snapshots every {saveEvery} steps (≈{saveEvery * dt:g}s)")

    nodes, dL = createRod(nv, L_sim)
    q0 = getStateVectors(nodes)

    antsObj = Ants(NV=nv, F0=F0_cluster, F_IND=F_IND,
                   NEST_DIR=np.array([1.0, 0.0]),
                   Kon=Kon, Koff=Koff, Kforget=Kforget,
                   Kconvert=Kconvert, Kreorient=Kreorient,
                   phiDamping=phiDamping, phiMax=phiMax)

    cTime, eventTime = 0.0, 0.0
    q = q0.copy()

    time_hist, q_hist, ants_hist, angles_hist = [], [], [], []
    step = 0
    pbar = tqdm(total=int(totalTime / dt), desc="Simulating", unit="steps")

    time_hist.append(cTime)
    q_hist.append(q.copy())
    ants_hist.append(antsObj.ants.copy())
    angles_hist.append(antsObj.angles.copy())

    while cTime < totalTime:
        if step % saveEvery == 0:
            time_hist.append(cTime)
            q_hist.append(q.copy())
            ants_hist.append(antsObj.ants.copy())
            angles_hist.append(antsObj.angles.copy())
        try:
            q, F, eventTime = solveStep(q, dL, nv, dt, gamma, EI, EA,
                                    antsObj, cTime, eventTime)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            print("Saving data before exiting...")
            break
        cTime += dt
        step += 1
        pbar.update(1)

    if saveData:
        dirname = datetime.datetime.now().strftime("sim_%Y%m%d")
        os.makedirs(dirname, exist_ok=True)
        if output is not None:
            fname = os.path.join(dirname, output)
        else:
            fname = os.path.join(dirname,
                             f"rod_L{L_sim:.1f}cm_seg{seg_len:.2f}cm.npz")
        np.savez(fname,
                 time=np.array(time_hist),
                 q=np.array(q_hist),
                 ants=np.array(ants_hist),
                 angles=np.array(angles_hist),
                 params=json.dumps({
                     "L_cm": L_sim,
                     "seg_len_cm": seg_len,
                     "nv": nv,
                     "EA": EA,
                     "EI": EI,
                     "gamma": gamma,
                     "F0": F0,
                     "F_IND": F_IND,
                     "Kon": Kon,
                     "Koff": Koff,
                     "Kforget": Kforget,
                     "Kconvert": Kconvert,
                     "Kreorient": Kreorient,
                     "dt": dt
                 }))
        print(f'Data saved → {fname}')
    pbar.close()


if __name__ == "__main__":
    t0 = time()
    Lrange = [5.0, 10.0, 15.0]
    for itr in range(100):
        Kon, Koff, Kforget, Kconvert, Kreorient = np.random.uniform(0.02, 1.0, 5)
        params = {
            'Kon': Kon,
            'Koff': Koff,
            'Kforget': Kforget,
            'Kconvert': Kconvert,
            'Kreorient': Kreorient
        }
        for L_idx in prange(3):
            L = Lrange[L_idx]
            run(totalTime=2000.0, L=L, params=params,
                seg_len=0.5, saveData=True,
                output=f"rod_L{L:.1f}cm_seg0.5cm_paramItr{itr:03d}.npz")
    print(f'Finished in {time() - t0:.1f}s')
