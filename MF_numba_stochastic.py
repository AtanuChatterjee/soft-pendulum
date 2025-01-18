### Mean-field code to be modified into stochastic microscopic model using the refactored ant class.

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from time import time

global nv, l
global dt, p, g, gamma, f0, kc, F_ind, Dp0, nestDir, EI, EA

l = 20

nv, dt = 5, 1e-4
p, g, gamma = 8.4, 1.6, 5.5 # 8.4, 1.6
f0, kc, F_ind, nestDir = 2.8, 0.7, 0.428, np.array([1, 0])
EI, EA = 5e3, 1e4


def plotrod(q, cTime):
    plt.clf()
    x = q[0:len(q):2]
    y = q[1:len(q):2]
    plt.plot(x, y, 'ko-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-l * 1.2, l * 1.2])
    plt.ylim([-l * 1.2, l * 1.2])
    plt.title(f'$t = {cTime:.1f}$')
    plt.tight_layout()
    plt.pause(0.001)


def createRod(Dp0):
    nodes = np.zeros((nv, 3))
    nodes[:, 0] = np.linspace(0, l, nv)
    nodes[1:, 2] = Dp0
    dL = np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1).reshape(-1, 1)
    ne = nv - 1
    return nodes, ne, dL


def getStateVectors(nodes):
    q0 = np.zeros((2 * nv, 1))
    q0[::2, 0] = nodes[:, 0]
    q0[1::2, 0] = nodes[:, 1]
    Dp0 = np.zeros((nv, 1))
    Dp0[1:, 0] = nodes[1:, 2]
    return q0, Dp0


@njit
def getFp(FOld, q, Dp0):
    Fp = np.zeros((2 * nv, 1))
    Dp = Dp0.copy()
    for k in range(nv - 1):
        Fk = np.ascontiguousarray(FOld[2 * k: 2 * k + 2, :])
        q_ = np.ascontiguousarray(q[2 * k: 2 * k + 2, :])
        q_normal = np.array([-q_[1, 0], q_[0, 0]])
        n = q_normal / (np.linalg.norm(q_normal) + 1e-34)
        n = np.ascontiguousarray(n[:, np.newaxis])
        dot_product = np.dot(n.T, Fk)
        Dp[k] = dt * (p * kc * np.sinh(dot_product / F_ind) - 2 * kc * Dp0[k] * np.cosh(dot_product / F_ind)) + Dp0[k]
        Fp[2 * k: 2 * k + 2] += f0 * Dp[k] * n
    return Fp, Dp



@njit
def getFg(q):
    Fg = np.zeros((2 * nv, 1))
    for k in range(nv - 1):
        Fg[2 * k:2 * k + 2, 0] += f0 * g * nestDir
    return Fg


@njit
def gradEs(q, refLen):
    n_nodes = q.shape[0] // 2
    F = np.zeros(q.shape)

    for i in range(n_nodes - 1):
        delta_x = q[2 * (i + 1)] - q[2 * i]
        delta_y = q[2 * (i + 1) + 1] - q[2 * i + 1]
        current_length = np.sqrt(delta_x ** 2 + delta_y ** 2)

        stretch = (EA / refLen[i]) * (1 - current_length / refLen[i])
        Fx = stretch * delta_x / refLen[i]
        Fy = stretch * delta_y / refLen[i]

        F[2 * i] -= Fx
        F[2 * i + 1] -= Fy
        F[2 * (i + 1)] += Fx
        F[2 * (i + 1) + 1] += Fy

    return np.ascontiguousarray(F.reshape(-1, 1))




@njit
def getFs(q, dL):
    refLen = dL
    F = gradEs(q, refLen) 
    return F


@njit
def gradEb(nodes, refLen):
    node0, node1, node2 = nodes[0:2], nodes[2:4], nodes[4:6]
    node0 = np.array([node0[0], node0[1], 0.0])
    node1 = np.array([node1[0], node1[1], 0.0])
    node2 = np.array([node2[0], node2[1], 0.0])

    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    te = ee / norm_e
    tf = ef / norm_f

    cross_prod = np.cross(te, tf)
    dot_prod = np.dot(te, tf)

    kb = 2.0 * cross_prod / (1.0 + dot_prod + 1e-34)
    kappa = kb[2]

    DkappaDe = 1.0 / norm_e * (-kappa * te + np.cross(tf, te) / (1.0 + dot_prod + 1e-34))
    DkappaDf = 1.0 / norm_f * (-kappa * te - np.cross(te, tf) / (1.0 + dot_prod + 1e-34))

    gradKappa = np.zeros(6)
    gradKappa[0:2] = -DkappaDe[0:2]
    gradKappa[2:4] = DkappaDe[0:2] - DkappaDf[0:2]
    gradKappa[4:6] = DkappaDf[0:2]

    F = gradKappa * EI * kappa / (refLen * refLen)
    return F.reshape(-1, 1)



@njit
def getFb(q, dL):
    Fb = np.zeros((2 * nv, 1))
    for k in range(nv - 2):
        nodes = q[2 * k: 2 * (k + 3)].flatten()
        gradEnergy = gradEb(nodes, dL[k])
        Fb[2 * k: 2 * (k + 3)] -= gradEnergy
    return Fb



@njit
def solve(q, q0, qDot, Dp0, dL):
    Fb = getFb(q, dL)
    Fs = getFs(q, dL)
    Fg = getFg(q)
    F = qDot
    Fp, Dp = getFp(F, q, Dp0)
    Ftot = Fp + Fg + Fs + Fb
    qDot = Ftot / gamma
    qNew = q0 + qDot * dt
    qNew[0: 2, 0] = q0[0: 2, 0]  # hinged
    # qNew[2: 4, 0] = q0[2: 4, 0]  # clamped
    return qNew, Dp


def run(totalTime, Dp0=0.1):
    nodes, ne, dL = createRod(Dp0)
    q0, Dp0 = getStateVectors(nodes)

    Nsteps = round(totalTime / dt)
    all_q = np.zeros((Nsteps, 2 * nv))

    q = q0
    Dp = Dp0
    qDot = (q - q0) / dt

    fig, ax = plt.subplots(figsize=(5, 5))

    cTime = 0
    for timeStep in range(1, Nsteps):
        # print(f't = {cTime}\n')
        q, Dp = solve(q, q0, qDot, Dp0, dL)
        qDot = (q - q0) / dt
        cTime += dt
        q0 = q
        Dp0 = Dp

        all_q[timeStep, :] = q.T
        continue
        if ((timeStep - 1) % 1e3 == 0):
            # continue
            plt.clf()
            plotrod(q, cTime)
            # plt.pause(0.01)
            # plt.savefig(f'./plots/rod_{str(timeStep).zfill(7)}.png')
    # np.savez('rod_data_30_hinged.npz', q=all_q)
    print('Done with execution!')

t = time()
run(totalTime=100)
print('Elapsed time:', time() - t)
