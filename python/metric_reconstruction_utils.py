import numpy as np
import cv2
import pickle

from utils import *
from scipy import optimize

def undistort(pd, d0, d1, p0):
    def fun(X, pd, d0, d1, p0):
        pn = X.reshape(2,1)
        diff = pn-p0
        r2 = np.inner(diff.T, diff.T)
        r4 = r2**2
        d = 1 + d0 * r2 + d1 * r4
        res = (pd-p0) / d + p0 - pn
        return res.reshape(2,)

    sol = optimize.root(fun, [pd[0], pd[1]], method='hybr', args=(pd, d0, d1, p0))
    return sol.x.reshape(2,1)


def compute_distortion(p, d0, d1, p0):
    diff = p-p0
    r2 = np.sum(diff**2,axis=0)
    r4 = r2**2
    pd = diff * (1 + r2*d0 + r4*d1) + p0
    #pd = p * (1 + r2 * d0 + r4 * d1)
    return pd


def unproject_to_world(p, K, dist):
    pc = np.array([[K[0, 2]], [K[1, 2]]])
    f = np.array([[K[0, 0]], [K[1, 1]]])
    if len(p) > 2:
        ph = np.array([p[0] / p[-1], p[1] / p[-1]]).reshape(2, 1)
    else:
        ph = p.reshape(2, 1)
    xd = undistort(ph, dist[0], dist[1], pc)
    xd = xd - pc
    xn = xd / f
    xn = np.vstack((xn, 1))
    return xn


def project_to_image(y, K, R = np.identity(3), T=0, dist = np.array([0,0])):
    y = y.reshape(3,1)
    yc = R @ y + T
    p = K @ yc
    p = p / p[-1]
    p = p[0:2]
    p = compute_distortion(p, dist[0], dist[1], np.array([K[0, 2], K[1, 2]]).reshape(2,1))
    return p[0:2]