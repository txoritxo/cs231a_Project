import numpy as np
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt
import cv2 as cv
from utils import *
from bundle_adjustment import *


def plot_packed_homographies(X, pdef, p, y_camera, camera_indices, total_observations, images):
    points, H,invH, d0, d1, p0x, p0y = unpack_parameters(X, pdef)
    added_image = images[0].im
    use_packed = True
    for i, idx in enumerate(camera_indices[1:], start=1):
        img = images[idx]
        curH = H[i] if use_packed else img.H
        h, w = img.im.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, curH)
        im_dst = cv.warpPerspective(img.im, curH, dsize=(w, h))
        added_image = cv.addWeighted(added_image, 0.6, im_dst, 0.3, 0)
        added_image = cv.polylines(added_image, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        cv.imshow("Image1", added_image)
        cv.waitKey(0)
        z = 0


def plot_pixel_errors(res0, res1):
    res0 = res0.reshape((2, -1))
    res1 = res1.reshape((2, -1))
    n0 = np.linalg.norm(res0, axis=0)
    n1 = np.linalg.norm(res1, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(res0[0,:], res0[1,:], 'rx', markersize=3)
    ax1.plot(res1[0, :], res1[1, :], 'bx', markersize=3)
    ax2.hist(n0, bins = 100, color = "skyblue")
    ax2.hist(n1, bins = 100, color = "red", alpha=0.5)
    plt.show()
    cv.waitKey(0)

def plot_packed_correspondences(X, pdef, p, y_camera, camera_indices, total_observations, images):
    img0 = images[0]
    points, H, invH, d0, d1, p0x, p0y = unpack_parameters(X, pdef)

    for i, cid in enumerate(camera_indices):
        pt0 = points[y_camera[cid]]
        pt0 = pt0/(pt0[:, -1].reshape((pt0.shape[0],1)))
        pt1 = p[cid]
        validate_packed_correspondences(img0, images[cid], pt0[:,0:2].T, pt1.T, H[i])


def plot_adjusted_correspondences(X, pdef, p, y_camera, camera_indices, total_observations, images):
    img0 = images[0]
    points, H, invH, d0, d1, p0x, p0y = unpack_parameters(X, pdef)
    pc = np.array([p0x, p0y]).reshape(2, 1)

    #1 project points in camera0 and apply distorsion
    p0 = invH[0] @ points.T
    p0 = p0/p0[-1, :]
    p0d = compute_distortion(p0[0:2, :], d0, d1, pc)
    p0d = p0d.T
    for i, cid in enumerate(camera_indices):
        pt0 = p0d[y_camera[cid]]

        p1 = invH[i] @ points[y_camera[cid]].T
        p1 = p1 / p1[-1, :]
        p1d = compute_distortion(p1[0:2, :], d0, d1, pc)
        p1d = p1d.T
        big_image = np.concatenate((images[cid].im, images[0].im), axis=1)
        plt.imshow(big_image, cmap='gray')
        h, w = images[0].im.shape
        pt0 = pt0.T
        p1d=p1d.T
        obs = p[cid]
        obs = obs.T
        for j in range(pt0.shape[1]):
            x0, y0 = pt0[0, j] + w, pt0[1, j]
            x1, y1 = p1d[0, j], p1d[1, j]
            plt.plot([x0, x1], [y0, y1], '-ro', linewidth=0.2, markersize=1)
            plt.plot(obs[0,j], obs[1,j], 'gx', linewidth=0.2, markersize=3)
        plt.show()
        cv.waitKey(0)



def validate_packed_correspondences(img0, img1, pt0, pt1, H):
    big_image = np.concatenate((img1.im, img0.im), axis=1)
    h, w = img0.im.shape
    plt.imshow(big_image, cmap='gray')
    pth0 = np.vstack((pt0, np.ones((1, pt0.shape[1]))))
    for i in range(pt0.shape[1]):
        x0, y0 = pt0[0,i]+w, pt0[1, i]
        x1, y1 = pt1[0, i], pt1[1, i]
        plt.plot([x0,x1],[y0,y1], '-ro', linewidth=0.2, markersize=1)

    use_packed_homography = True
    curH = H if use_packed_homography else img1.H
    pth1 = np.linalg.inv(curH) @ pth0
    pth1 = pth1/pth1[-1]

    qpt1h = np.vstack((pt1, np.ones((1, pt1.shape[1]))))
    pth00 = curH @ qpt1h
    pth00 = pth00/pth00[-1]
    res = np.linalg.norm(pth1[0:2,:] - pt1, axis=0)
    for i in range(pt0.shape[1]):
        x1, y1 = pth1[0, i], pth1[1, i]
        x2, y2 = pth00[0, i]+w, pth00[1, i]
        plt.plot(x1, y1, 'bx', linewidth=0.1, markersize=3)
        plt.plot(x2, y2, 'bx', linewidth=0.1, markersize=3)
    plt.show()
    cv.waitKey(0)
