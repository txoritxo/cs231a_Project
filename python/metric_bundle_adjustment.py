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
        R = Rmats[i]
        T = Tvecs[i]
        proj = np.zeros((p3d.shape[0],2))
        for j in range(p3d.shape[0]):
            proj[j] = project_to_image(p3d[j], K, R, T, dist).T
        diff = (observations[cam]-proj).ravel()
        residual[last_index:last_index+len(diff)] = diff
        last_index += len(diff)
    return residual
