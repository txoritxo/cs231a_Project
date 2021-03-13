import numpy as np
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt
import cv2 as cv
from utils import *
from metric_reconstruction_utils import *

class problem_definition:
    def __init__(self):
        self.rot_shape = None
        self.rot_len = None
        self.t_shape = None
        self.t_len = None
        self.p3d_shape = None
        self.p3d_len = None

def unpack_parameters(X, p:problem_definition):
    K = np.array([
        [X[0], 0, X[2]],
        [0, X[1], X[3]],
        [0,   0,    1]
    ])
    d = np.array([X[4], X[5]]).reshape(2,1)
    rotMats = np.zeros(p.rot_shape)
    for i in range(int(p.rot_len/3)):
        cur_index = 6 + i*3
        rotMats[i], _ = cv2.Rodrigues(X[cur_index:cur_index+3])
    cur_id = 6 + p.rot_len
    tvecs = X[cur_id: cur_id + p.t_len].reshape(p.t_shape)
    cur_id += p.t_len
    p3d = X[cur_id:cur_id+p.p3d_len].reshape(p.p3d_shape)
    p3d = np.hstack((p3d, np.zeros((p3d.shape[0], 1))))
    return K, d, rotMats, tvecs, p3d

def unpack_rodrigues_parameters(X, p:problem_definition):
    rotvecs = np.zeros((int(p.rot_len/3), 3))
    for i in range(int(p.rot_len/3)):
        cur_index = 6 + i*3
        rotvecs[i] = X[cur_index:cur_index+3]
    return rotvecs

#y0, p, H, y_camera, camera_indices
def pack_parameters(K, d, rotMats, tvecs, p3d):
    pdef = problem_definition()
    pdef.rot_shape = rotMats.shape
    pdef.t_shape = tvecs.shape
    nvars = 6 + len(rotMats)*3 + len(tvecs)*3 + len(p3d)*2
    X = np.zeros((nvars))
    X[0] = K[0, 0]
    X[1] = K[1, 1]
    X[2] = K[0, 2]
    X[3] = K[1, 2]
    X[4] = d[0]
    X[5] = d[1]
    cur_id = 6
    for i in range(len(rotMats)):
        cur_id = 6 + i*3
        X[cur_id:cur_id+3] = cv2.Rodrigues(rotMats[i])[0].T
    cur_id += 3
    flat_t = tvecs.flatten()
    X[cur_id: cur_id+len(flat_t)] = flat_t
    cur_id += len(flat_t)
    p3d = p3d[:, 0:2]
    pdef.p3d_shape = p3d.shape
    p3dflat = p3d.flatten()
    X[cur_id: cur_id + len(p3dflat)] = p3dflat
    pdef.rot_len = len(rotMats)*3
    pdef.t_len = len(flat_t)
    pdef.p3d_len = len(p3dflat)
    return X, pdef


def compute_residual(X, pdef, observations, y_camera, camera_indices, total_observations):
    K, dist, Rmats, Tvecs, p3Dref = unpack_parameters(X, pdef)

    residual = np.zeros(total_observations * 2)
    last_index = 0
    for i, cam in enumerate(camera_indices):
        p3d = p3Dref[y_camera[cam]]
        R = Rmats[cam]
        T = Tvecs[cam]
        proj = np.zeros((p3d.shape[0],2))
        for j in range(p3d.shape[0]):
            proj[j] = project_to_image(p3d[j], K, R, T, dist).T
        diff = (observations[cam]-proj).ravel()
        residual[last_index:last_index+len(diff)] = diff
        last_index += len(diff)
    return residual


def compute_mba_Jacobian(X, pdef, observations, y_camera, camera_indices, total_observations):
    K, dist, Rmats, Tvecs, p3Dref = unpack_parameters(X, pdef)
    rotvecs = unpack_rodrigues_parameters(X, pdef)
    p0 = K[:2,2]
    jac = np.zeros((total_observations*2, len(X)))
    startyr = 6
    startyt = 6 + 3*len(camera_indices)
    startyx = 6 + 6*len(camera_indices)

    for i, cam in enumerate(camera_indices):
        indy = y_camera[cam]
        p3d = p3Dref[indy]
        R = Rmats[cam]
        T = Tvecs[cam]
        z1 = p3d @ R.T + T.T
        z2 = z1/z1[:,2].reshape(len(z1),1)
        z3 = z2 @ K.T[:,:2]
        z2 = z2[:,:2]
        r2 = np.sum((z3-p0.reshape(1,2))**2,axis=1)

        for j in range(len(indy)):
            dz1dx = R[:, :2]  # z1 = R x + t
            dz2dz1 = np.hstack((np.eye(2), -z2[j].reshape(2, 1))) / z1[j][2]  # z2 = z1[:2] / z1[2]
            dz3dz2 = K[:2,:2]        # z3 = K z2
            dr2dz3 = 2 * (z3[j] - p0).reshape(1, 2)  # r2 = (z3-p0).T (z3-p0)
            dz4dr2 = (dist[0] + 2*r2[j] * dist[1]) * (z2[j] - p0).reshape(2, 1)  # z4 = (z3-p0) (1+d0*r2+d1*r4) + p0
            dz4dz3 = (1 + dist[0] * r2[j] + dist[1] * r2[j] ** 2) * np.eye(2)
            dXdx = -(dz4dr2 @ dr2dz3 @ dz3dz2 @ dz2dz1 @ dz1dx +
                     dz4dz3 @ dz3dz2 @ dz2dz1 @ dz1dx)
            rangex = np.array(range(2*indy[j], 2*indy[j]+2))
            jac[rangex][:,np.array(startyx+np.array(rangex))] = dXdx
            dz1dr = rodrigues_rotation_derivative(rotvecs[i], p3d[j])
            dXdr = -(dz4dr2 @ dr2dz3 @ dz3dz2 @ dz2dz1 @ dz1dr +
                     dz4dz3 @ dz3dz2 @ dz2dz1 @ dz1dr)
            rangeyr = np.array(range(startyr+3*i, startyr+3*(i+1)))
            jac[rangex][:,rangeyr] = dXdr
            dz1dt = np.eye(3)
            dXdt = -(dz4dr2 @ dr2dz3 @ dz3dz2 @ dz2dz1 @ dz1dt +
                     dz4dz3 @ dz3dz2 @ dz2dz1 @ dz1dt)
            rangeyt = np.array(range(startyt+3*i, startyt+3*(i+1)))
            jac[rangex][:,rangeyt] = dXdt
            dz4dd = np.hstack((r2[j]*(z3[j]-p0).reshape(2,1), (r2[j]**2)*(z3[j]-p0).reshape(2,1)))
            dXdd = -dz4dd
            jac[rangex, 4:6] = dXdd
            dr2dp0 = -2*(z3[j]-p0).reshape(1,2)
            dz4dp0 = (1-(1 + dist[0]*r2[j]+dist[1]*r2[j]**2))*np.eye(2)
            dz3dp0 = np.eye(2)
            dXdp0 = -(dz4dr2 @ dr2dp0 + dz4dr2 @ dr2dz3 @ dz3dp0 +
                      dz4dp0 + dz4dz3 @ dz3dp0)
            jac[rangex, 2:4] = dXdp0
            dz3df = np.diag(z2[j])
            dXdf = -(dz4dr2 @ dr2dz3 @ dz3df + dz4dz3 @ dz3df)
            jac[rangex, :2] = dXdf

    return jac
