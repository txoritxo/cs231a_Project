import numpy as np
import cv2
import pickle
from bundle_adjustment import *

def metric_reconstruction(y0, p, y_camera, camera_indices,K,a,b,n,d0,d1):
    #Compute initial R and t for the reference frame
    R = np.hstack((a,b,n))
    R = R.T
    t = -np.matmul(R,n)
    refCenter = -t

    #Compute 3D coordinates of the feature points in the reference frame
    P3Dref = []
    dist = np.array([d0, d1, 0, 0])
    for pi in y0:
        xn = np.array([pi[0]*K[0,0]+K[0,2],pi[1]*K[1,1]+K[1,2], 1])
        xn = cv2.undistort(xn,K,dist)
        xn = xn[:,0] ##missing applying distortion
        xdir = np.matmul(R.T,xn.T)
        xdir=xdir.reshape((3,1))
        # if xdir[2]>0:
        #     continue
        # else:
        #    xdir[2] = 0.0001
        Pos3Dref = refCenter - (refCenter[2] / xdir[2]) * xdir
        Pos3Dref[2] = 0
        P3Dref.append(Pos3Dref.T)

    P3Dref = np.array(P3Dref)

    #Compute the extrinsics for the rest of selected frames
    Rmats = [R]
    Tvecs = [t]
    for i, cam_index in enumerate(camera_indices):
        if cam_index > 0:
            Pref = P3Dref[y_camera[cam_index]]
            Pref = Pref[:,0,:]
            Pimg = p[cam_index].astype(np.float32)
            ret, rvec, tvec, _ = cv2.solvePnPRansac(Pref, Pimg, K, distCoeffs=None)
            Rotmat, jacobian = cv2.Rodrigues(rvec)
            Rmats.append(Rotmat)
            Tvecs.append(tvec)
    Rmats = np.array(Rmats)
    Tvecs = np.array(Tvecs)

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

    ##Testing
    Pref = P3Dref[y_camera[1]]
    Pref = Pref[1, 0, :]
    points = p[1]
    p_est = np.matmul(np.matmul(K,Rmats[1]),Pref.T)+Tvecs[1].T
    p_est = p_est[0]
    p_est = p_est/p_est[2]
    print('Estimated point',p_est)
    print('Actual point',points[1])




