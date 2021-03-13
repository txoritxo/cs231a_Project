import numpy as np
import cv2
import pickle

from utils import *
from homography_based_calibration import *
from scipy.optimize import NonlinearConstraint
from scipy import optimize
from metric_reconstruction_utils import *
import cv2

def print_reprojections(cam_id, p3d, y_camera, p, K, dist, R, t):
    for i, id in enumerate(y_camera[cam_id]):
        observation = p[cam_id][i]
        scene_point_id = y_camera[cam_id][i]
        scene_point = p3d[scene_point_id]
        projection = project_to_image(scene_point, K, R, t, dist)
        print('i: {}; exp:[{:5.1f}, {:5.1f}]; act:[{:5.1f}, {:5.1f}]'.format(i, np.asscalar(observation[0]),
                                                                             np.asscalar(observation[1]),
                                                                             np.asscalar(projection[0]),
                                                                             np.asscalar(projection[1])))


def print_reprojections_opencv(p3d, p, rotvec, Rotmat, tvec, K, dist=None):
    proj, jacobian = cv2.projectPoints(p3d, rotvec, tvec, K, dist)

    for i in range(proj.shape[0]):
        observation = p[i]
        cvprojection = proj[i].reshape(2,1)
        scene_point = p3d[i]
        projection = project_to_image(scene_point, K, Rotmat, tvec)

        print('i: {}; exp:[{:5.1f}, {:5.1f}]; actcv:[{:5.1f}, {:5.1f}]; act:[{:5.1f}, {:5.1f}]'.format(
            i, np.asscalar(observation[0]), np.asscalar(observation[1]),
            np.asscalar(cvprojection[0]), np.asscalar(cvprojection[1]),
            np.asscalar(projection[0]), np.asscalar(projection[1])))


def compute_R_t(n,a,b):
    R = np.hstack((a, b, n))
    #R = R.T
    t = - R @ n
    return R, t


def metric_reconstruction(y0, p, y_camera, camera_indices,K,a,b,n,d0,d1):
    #Compute initial R and t for the reference frame
    R, t = compute_R_t(n, a, b)
    refCenter = n

    #Compute 3D coordinates of the feature points in the reference frame
    p3Dref = []
    dist = np.array([d0, d1, 0, 0])
    for pi in y0:
        xn = unproject_to_world(pi, K, dist)
        xdir = R.T @ xn
        pos3D = refCenter - (refCenter[2]/xdir[2]) * xdir
        pos3D[2] = 0
        p3Dref.append(pos3D.T)

    p3Dref = np.array(p3Dref)
    p3Dref = p3Dref.reshape(p3Dref.shape[0], p3Dref.shape[2])

    #Compute the extrinsics for the rest of selected frames
    Rmats = [R]
    Tvecs = [t]
    for i, cam_index in enumerate(camera_indices):
        if cam_index > 0:
            Pref = p3Dref[y_camera[cam_index]]
            Pimg = p[cam_index].astype(np.float32)
            ret, rvec, tvec, _ = cv2.solvePnPRansac(Pref, Pimg, K, distCoeffs=None)
            Rotmat, jacobian = cv2.Rodrigues(rvec)
            Rmats.append(Rotmat)
            Tvecs.append(tvec)
            print_reprojections_opencv(Pref, Pimg, rvec, Rotmat, tvec, K)
    Rmats = np.array(Rmats)
    Tvecs = np.array(Tvecs)
    return p3Dref, Rmats, Tvecs

    # ##Using real K
    # K_real = np.load('/media/BRTE-mpalomar/3A7A03687A03206D/SkyData/CameraMatrix.npy')
    # Rmats_real = [R]
    # Tvecs_real = [t]
    # for i, cam_index in enumerate(camera_indices):
    #     if cam_index > 0:
    #         Pref = P3Dref[y_camera[cam_index]]
    #         Pref = Pref[:,0,:]
    #         Pimg = p[cam_index].astype(np.float32)
    #         ret, rvec, tvec, _ = cv2.solvePnPRansac(Pref, Pimg, K_real, distCoeffs=None)
    #         Rotmat, jacobian = cv2.Rodrigues(rvec)
    #         Rmats_real.append(Rotmat)
    #         Tvecs_real.append(tvec)
    # Rmats_real = np.array(Rmats)
    # Tvecs_real = np.array(Tvecs)




