import numpy as np
import cv2
import pickle

from utils import *
from scipy import optimize

# def undistort(pd, d0, d1, p0):
#     def fun(X, pd, d0, d1, p0):
#         pn = X.reshape(2,1)
#         diff = pn-p0
#         r2 = np.inner(diff.T, diff.T)
#         r4 = r2**2
#         d = 1 + d0 * r2 + d1 * r4
#         res = (pd-p0) / d + p0 - pn
#         return res.reshape(2,)
#
#     sol = optimize.root(fun, [pd[0], pd[1]], method='hybr', args=(pd, d0, d1, p0))
#     return sol.x.reshape(2,1)

def undistort(pd, d0, d1, p0):
    # Find roots for d1 rn**5 + d0 rn**3 + rn - rd = 0
    rd = np.linalg.norm(pd - p0)
    rn_opts = np.roots(np.array([d1, 0, d0, 0, 1.0, -rd]))
    rn_opts = np.real(rn_opts[np.imag(rn_opts)==0.0])
    rn = np.maximum(0, np.minimum(rd, rn_opts[np.argmin(np.abs(rn_opts - rd))]))
    x = (pd - p0) * rn / rd + p0
    return x


def compute_distortion(p, d0, d1, p0):
    diff = p-p0
    r2 = np.sum(diff**2,axis=0)
    r4 = r2**2
    pd = diff * (1 + r2*d0 + r4*d1) + p0
    #pd = p * (1 + r2 * d0 + r4 * d1)
    return pd


def unproject_to_world(p, K, dist):
    pc = K[:2, 2].reshape(2,1)
    f = np.array([[K[0, 0]], [K[1, 1]]])
    if len(p) > 2:
        ph = np.array([p[0] / p[-1], p[1] / p[-1]]).reshape(2, 1)
    else:
        ph = p.reshape(2, 1)
    xd = undistort(ph, dist[0], dist[1], pc)
    xn = (xd - pc) / f
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


def dcm_rodrigues(r):
    # DCM_RODRIGUES
    # Creates the rotation matrix from the input Rodrigues vector
    #
    # Inputs:
    # - r is a 1x3 vector containing a Rodrigues representation of a rotation.
    # Outputs:
    # - R [3x3] is the rotation matrix.
    theta = np.linalg.norm(r)
    r_nrm = r / theta
    R = np.cos(theta)*np.eye(3) + np.sin(theta)*skew(r_nrm) + (1-np.cos(theta))*np.outer(r_nrm, r_nrm)
    return R


def rodrigues_dcm(R):
    # DCM_RODRIGUES
    # Creates the rotation matrix from the input Rodrigues vector
    #
    # Inputs:
    # - R [3x3] is the rotation matrix.
    # Outputs:
    # - r is a 1x3 vector containing a Rodrigues representation of a rotation.
    theta = np.arccos((np.trace(R)-1)/2)
    if np.abs(theta) > 1e-12:
        r_nrm = (1./(2*np.sin(theta))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0],R[1,0]-R[0,1]])
    else:
        r_nrm = np.array([1., 0., 0.])
    r = theta * r_nrm
    return r


def rodrigues_rotate_direct(r, vecin):
    # RODRIGUES_ROTATE_DIRECT
    # Takes the input vector (vecin) and rotates it by the input Rodrigues vector to
    # obtain the output vector vecout.
    #
    # Inputs:
    # - r is a 1x3 vector containing a Rodrigues representation of a rotation.
    # - vecin [1x3] is the input vector.
    #
    # Outputs:
    # - vecout [1x3] is the rotated input vector.
    theta = np.linalg.norm(r)
    r_nrm = r / theta
    R = np.cos(theta)*np.eye(3) + np.sin(theta)*skew(r_nrm) + (1-np.cos(theta))*np.outer(r_nrm, r_nrm)
    return R @ vecin

def rodrigues_rotate_vector(rot_vecs, points):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


##
# @brief calculate derivative of rotation of an arbitrary vector wrt Rodrigues parameters
def rodrigues_rotation_derivative(r, u):
    # See "A compact formula for the derivative of a 3-D rotation in  exponential coordinates"
    # Inputs:
    # - r is a 1x3 vector Rodrigues parameter vector
    # - vecin [1x3] is the input vector to be rotated.
    # Outputs:
    # - dRudv [3x3] is the derivative matrix
    R = dcm_rodrigues(r)
    dRudv = -R @ skew(u) @ (np.outer(r, r) + (R.transpose() - np.eye(3)) @ skew(r)) / np.linalg.norm(r)**2
    return dRudv


##
# @brief transform 3-vector to skew symmetric 3x3 matrix
def skew(om):
    return np.array([[0, -om[2], om[1]], [om[2], 0, -om[0]], [-om[1], om[0], 0]])

