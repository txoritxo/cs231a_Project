import numpy as np
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt
import cv2 as cv
from utils import *

class problem_definition:
    def __init__(self):
        self.point_indices = []
        self.H_indices = []
        self.p0_indices = []
        self.d0_indices = []
        self.ncameras = 0
        self.fix_camera0 = True

def unpack_parameters(X, p:problem_definition):
    points = X[p.point_indices[0]:p.point_indices[-1]].reshape(-1, 2)
    xh = X[p.point_indices[-1]]
    points = np.hstack((points, np.ones((points.shape[0], 1))*xh))
    iter_cameras = (p.ncameras - 1) if p.fix_camera0 else p.ncameras
    H = np.zeros((p.ncameras, 3, 3))
    invH = np.zeros((p.ncameras, 3, 3))
    H[0] = np.identity(3)
    invH[0] = np.identity(3)
    last_index = p.point_indices[-1]+1
    for i in range(iter_cameras):
        start_index = 8*i+last_index
        H[i + int(p.fix_camera0) ] = np.hstack((X[start_index:start_index+8], 1)).reshape(3,3)
        invH[i + int(p.fix_camera0)] = np.linalg.inv(H[i+ int(p.fix_camera0)])
    last_index += iter_cameras*8
    d0, d1, p0x, p0y = X[last_index:last_index+4]
    return points, H, invH, d0, d1, p0x, p0y


#y0, p, H, y_camera, camera_indices
def pack_parameters(points, H, d0, d1, p0x, p0y, fix_camera0=True):
    pdef = problem_definition()
    ncameras = H.shape[0]
    iter_cameras = (ncameras - int(fix_camera0))
    pdef.ncameras = ncameras
    pdef.fix_camera0 = fix_camera0
    sz = len(points)*2 + iter_cameras * 8 + 4 +1
    X = np.zeros(sz)
    p = points.flatten()
    X[0:p.shape[0]] = p
    X[p.shape[0]] = 1 # third world coordinate of the point. the same for all points
    last_index = p.shape[0]+1
    pdef.point_indices = [0, p.shape[0]]
    pdef.H_indices = [last_index, last_index + iter_cameras * 8]
    for i in range(iter_cameras):
        start_index = i*8 + last_index
        X[start_index:start_index+8] = H[i + int(fix_camera0)].flatten()[0:8]
    last_index += iter_cameras*8
    X[last_index:last_index+4] = np.array([d0, d1, p0x, p0y])
    pdef.d0_indices = [last_index, last_index+2]
    pdef.p0_indices = [last_index+2, last_index+4]
    return X, pdef


def compute_distortion(p, d0, d1, p0):
    diff = p-p0
    r2 = np.sum(diff**2,axis=0)
    r4 = r2**2
    pd = diff * (1 + r2*d0 + r4*d1) + p0
    return pd



def compute_residual(X, pdef, observations, y_camera, camera_indices, total_observations):
    points, H, invH, d0, d1, p0x, p0y = unpack_parameters(X, pdef)
    ##print('Trying with d= [%4.3f, %4.3f] and p0 = [%5.1f, 5.1f]', d0, d1, p0x, p0y)
    p0 = np.array([p0x, p0y]).reshape(2, 1)
    residual = np.zeros(total_observations*2)
    last_index = 0
    use_invH = True
    for i, cam_index in enumerate(camera_indices):
        y = points[y_camera[cam_index]]
        curH = invH[i] if use_invH else H[i]
        p = curH @ y.T
        p = p/p[-1, :]
        pd = compute_distortion(p[0:2, :], d0, d1, p0)
        diff = (observations[cam_index].T-pd).ravel()
        residual[last_index:last_index+len(diff)] = diff
        last_index += len(diff)
    return residual
