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
        self.p_cen = np.zeros(2)  # image geometrical center
        self.LAMBDA_SQRT = 1
        self.fix_camera0 = True

def unpack_parameters(X, p:problem_definition):
    points = X[p.point_indices[0]:p.point_indices[-1]+1].reshape(-1, 2)
    #xh = X[p.point_indices[-1]]
    #points = np.hstack((points, np.ones((points.shape[0], 1))*xh))
    iter_cameras = (p.ncameras - 1) if p.fix_camera0 else p.ncameras
    H = np.zeros((p.ncameras, 3, 3))
    invH = np.zeros((p.ncameras, 3, 3))
    H[0] = np.identity(3)
    invH[0] = np.identity(3)
    last_index = p.point_indices[-1]+1
    for i in range(iter_cameras):
        start_index = 8*i+last_index
        invH[i + int(p.fix_camera0) ] = np.hstack((X[start_index:start_index+8], 1)).reshape(3,3)
        H[i + int(p.fix_camera0)] = np.linalg.inv(invH[i+ int(p.fix_camera0)])
    last_index += iter_cameras*8
    d0, d1, p0x, p0y = X[last_index:last_index+4]
    return points, H, invH, d0, d1, p0x, p0y


#y0, p, H, y_camera, camera_indices
def pack_parameters(points, invH, d0, d1, p0x, p0y, p_cen, fix_camera0=True):
    pdef = problem_definition()
    ncameras = invH.shape[0]
    iter_cameras = (ncameras - int(fix_camera0))
    pdef.ncameras = ncameras
    pdef.fix_camera0 = fix_camera0
    sz = len(points)*2 + iter_cameras * 8 + 4
    X = np.zeros(sz)
    p = points.flatten()
    X[0:p.shape[0]] = p
    #X[p.shape[0]] = 1 # third world coordinate of the point. the same for all points
    last_index = p.shape[0]
    pdef.point_indices = [0, p.shape[0]-1]
    pdef.H_indices = [last_index, last_index + iter_cameras * 8]
    for i in range(iter_cameras):
        start_index = i*8 + last_index
        X[start_index:start_index+8] = invH[i + int(fix_camera0)].flatten()[0:8]
    last_index += iter_cameras*8
    X[last_index:last_index+4] = np.array([d0, d1, p0x, p0y])
    pdef.d0_indices = [last_index, last_index+2]
    pdef.p0_indices = [last_index+2, last_index+4]
    pdef.p_cen = p_cen
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
    p0 = np.array([p0x, p0y])
    residual = np.zeros(total_observations*2 + 2)
    last_index = 0
    for i, cam_index in enumerate(camera_indices[1:]):
        y = points[y_camera[cam_index]]
        p = cv.perspectiveTransform(y[:,:2].reshape(-1, 1, 2), invH[cam_index]).reshape(-1,2)
        # p = curH @ y.T
        # p = p/p[-1, :]
        pd = compute_distortion(p.T, d0, d1, p0.reshape(2, 1))
        diff = (observations[cam_index].T-pd).ravel()
        residual[last_index:last_index+len(diff)] = diff
        last_index += len(diff)
    residual[last_index:last_index+2] = pdef.LAMBDA_SQRT * (p0 - pdef.p_cen)
    return residual


def compute_Jacobian(X, pdef, observations, y_camera, camera_indices, total_observations):
    points, H, invH, d0, d1, p0x, p0y = unpack_parameters(X, pdef)
    ##print('Trying with d= [%4.3f, %4.3f] and p0 = [%5.1f, 5.1f]', d0, d1, p0x, p0y)
    p0 = np.array([p0x, p0y])
    jac = np.zeros((total_observations*2 + 2, len(X)))
    last_index = 0
    num_cams = len(camera_indices[1:])
    for i, cam_index in enumerate(camera_indices[1:]):
        indy = y_camera[cam_index]
        y = points[indy]
        z1 = np.column_stack((y, np.ones(len(y)))) @ invH[cam_index].T
        z2 = z1[:,:2]/z1[:,2].reshape(len(z1),1)
        #pd = compute_distortion(z2.T, d0, d1, p0.reshape(2, 1))
        r2 = np.sum((z2-p0.reshape(1,2))**2,axis=1)

        for j in range(len(indy)):
            dz1dy = invH[cam_index][:,:2]                                           # z1 = H y
            dz2dz1 = np.hstack((np.eye(2), -z2[j].reshape(2,1)))/z1[j][2]      # z2 = z1[:2] / z1[2]
            dr2dz2 = 2 * (z2[j]-p0).reshape(1,2)                                    # r2 = (z2-p0).T (z2-p0)
            dz3dr2 = (d0+2*r2[j]*d1)*(z2[j]-p0).reshape(2,1)                        # z3 = (z2-p0) (1+d0*r2+d1*r4) + p0
            dz3dz2 = (1 + d0*r2[j]+d1*r2[j]**2)*np.eye(2)
            dXdy = -(dz3dr2 @ dr2dz2 @ dz2dz1 @ dz1dy + dz3dz2 @ dz2dz1 @ dz1dy)
            jac[2*indy[j]:2*indy[j]+2, 2*indy[j]:2*indy[j]+2] = dXdy
            dz1dh = np.zeros((3,8))
            dz1dh[0,:3] = np.hstack((y[j], [1]))
            dz1dh[1,3:6] = np.hstack((y[j], [1]))
            dz1dh[2,6:] = y[j]
            dXdh = -(dz3dr2 @ dr2dz2 @ dz2dz1 @ dz1dh + dz3dz2 @ dz2dz1 @ dz1dh)
            indH = list(range(len(points)*2+i*8,len(points)*2+(i+1)*8))
            jac[2*indy[j]:2*indy[j]+2, indH] = dXdh
            dz3dd = np.hstack((r2[j]*(z2[j]-p0).reshape(2,1), (r2[j]**2)*(z2[j]-p0).reshape(2,1)))
            dXdd = -dz3dd
            indd = list(range(len(points)*2+num_cams*8,len(points)*2+num_cams*8+2))
            jac[2 * indy[j]:2 * indy[j] + 2, indd] = dXdd
            dr2dp0 = -2*(z2[j]-p0).reshape(1,2)
            dz3dp0 = (1-(1 + d0*r2[j]+d1*r2[j]**2))*np.eye(2)
            dXdp0 = -(dz3dr2 @ dr2dp0 + dz3dp0)
            indp0 = list(range(len(points)*2+num_cams*8+2,len(points)*2+num_cams*8+4))
            jac[2 * indy[j]:2 * indy[j] + 2, indp0] = dXdp0

    dXdp0 = 2*(p0-pdef.p_cen)
    jac[total_observations*2:, -2:] = dXdp0
    return jac


def loss_function_cauchy(z):
    # cauchy loss =
    F1 = np.vstack((np.log(1+z[:-2]), 1/(1+z[:-2]), -1/(1+z[:-2])**2))
    F2 = np.vstack((z[-2:], np.ones(2), np.zeros(2)))
    F = np.hstack((F1, F2))
    return F

