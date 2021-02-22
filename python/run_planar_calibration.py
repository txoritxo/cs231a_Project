import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.optimize as opt
from collections import defaultdict
#from scipy.optimize import newton_krylov
from utils import *
from run_bundle_adjustment import *
from bundle_adjustment import *
from run_homography_based_calibration import *

IMAGE_WIDTH = 800

class qimage:
    def __init__(self, filename):
        self.im = cv.imread(filename, 0)
        h, w = self.im.shape

        if h>w:
            self.im = cv.rotate(self.im, cv.ROTATE_90_CLOCKWISE)
            h, w = self.im.shape

        factor = w/h if w > h else h
        new_height = int(IMAGE_WIDTH/factor)
        #new_height = 480
        self.im = cv.resize(self.im, dsize=(IMAGE_WIDTH, new_height))
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        self.kp, self.des = sift.detectAndCompute(self.im, None)
        self.H = []
        self.mask = None
        self.good = None
        self.p0 = None
        self.p1 = None

def compute_homography(img1:qimage, img2:qimage):
    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #search_params = dict(checks=50)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(img1.des, img2.des, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([img1.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([img2.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        img1.good = good.copy()
        img1.mask = matchesMask.copy()
        img1.H = M
        #qplot_correspondences2(img1, img2, img1.good, img1.mask)
        return img1


def get_image_correspondences(img0, img1):
    p0 = np.array([], dtype=float).reshape(2, 0)
    p1 = np.array([], dtype=float).reshape(2, 0)
    for match, mask in zip(img1.good, img1.mask):
        if not mask:
            continue
        this_p0 = img0.kp[match.trainIdx].pt
        this_p1 = img1.kp[match.queryIdx].pt
        p0 = np.hstack((p0, np.array([[this_p0[0]], [this_p0[1]]])))
        p1 = np.hstack((p1, np.array([[this_p1[0]], [this_p1[1]]])))
    return p0, p1


def qload_images(idx0, lastidx):
    imgs = list()
    for i in range(idx0, lastidx):
        filename = '../data/quere/x-e3_18mm/photosPicture1/DSCF{}.jpg'.format(i)
        print('loading image {}'.format(filename))
        imgs.append(qimage(filename))
    print('STEP #1 - Loading images and identifying feature points :COMPLETED')
    return imgs


def qcompute_homographies(imgs):
    imgs[0].H = np.identity(3)
    print('STEP #2 - Computing Homographies')
    for i, img in enumerate(imgs[1:], start=1):
        img = compute_homography(img, imgs[0])
        img.p0, img.p1 = get_image_correspondences(imgs[0], img)
    return imgs


class Woint:
    def __init__(self):
        self.nobservations = 0
        self.cameras = list()

def qfilter_matches(imgs, min_matches=7):
    dict = defaultdict(int)
    for im in imgs[1:]:
        for match, mask in zip(im.good, im.mask):
            if not mask: continue
            dict[match.trainIdx] +=1
    dict2 = {key: val for key, val in dict.items() if val > min_matches}
    dict3 = {key: [0] for key in dict2}
    y_camera = [[] for i in range(len(imgs))]
    y0 = None
    p = [None]*len(imgs)

    for pt_index, id in enumerate(dict2):
        if p[0] is None:
            p[0] = np.array(imgs[0].kp[id].pt)
        else:
            p[0] = np.vstack((p[0], np.array(imgs[0].kp[id].pt)))
        y_camera[0].append(pt_index)
        for i, im in enumerate(imgs[1:], start=1):
            for match, mask in zip(im.good, im.mask):
                if not mask: continue
                if match.trainIdx == id:
                    dict3[id].append(i)
                    y_camera[i].append(pt_index)
                    if p[i] is None:
                        p[i] = np.array(im.kp[match.queryIdx].pt)
                    else:
                        p[i] = np.vstack((p[i], im.kp[match.queryIdx].pt))

    camera_indices = list(i for i, ob in enumerate(p) if len(ob) > 10)
    y0 = p[0].copy()
    H = np.zeros((len(camera_indices), 3, 3))
    for i, cam_id in enumerate(camera_indices):
        temp = imgs[cam_id].H/imgs[cam_id].H[-1, -1]
        H[i] = temp

    # camera indices: includes indices of cameras with more than 10 observations
    # p. It is a list of numpy arrays. The ith element of the list contains the ordered observations in camera i
    # y_camera. it is a list of list of indices. The ith element of the list contains indices of the 3d points
    # which have correspondence observations in camera i
    # y0 is a numpy array containing the positions of the points in world coordinates, in this case, for initialization
    # purposes it is the same as the observations from camera 0
    # H is a numpy array of size [ncameras, 3, 3] which contains the stacked homography matrices
    return y0, p, H, y_camera, camera_indices



def run_planar_calibration():
    #1 Set suffixes of pictures to load
    idx0 = 1975
    #idx0 = 1986
    lastidx = 2001

    # STEP #1. Feature Matching
    imgs = qload_images(idx0, lastidx) # load images and compute features
    # STEP #2. Projective Reconstruction
    imgs = qcompute_homographies(imgs) # compute homographies with respect to image[0]
    # filter matches. We don't want to keep points that are only seen by a few cameras
    # the minimum matches considered are given by parameter min_matches
    y0, p, H, y_camera, camera_indices = qfilter_matches(imgs, min_matches=7)

    # this function is for visual validation purposes, it'll draw point correspondences for all considered images in
    # camera_indices. such point correspondences will be extracted from the output of qfilter_matches
    #qtest_ba_parameter_extraction(imgs, y0, p, H, y_camera, camera_indices)
    #plot_homographies(imgs)
    #qprojective_ba(imgs, the_camera)

    # set initial camera center and distortion
    p0y, p0x = imgs[0].im.shape[0]/2, imgs[0].im.shape[1]/2
    d0, d1 = 0, 0

    # STEP #3 Projective Bundle Adjustment (4.2)
    opoints, oH, oinvH, od0, od1, op0x, op0y = run_bundle_adjustment(y0, p, H, y_camera, camera_indices, imgs)
    to_file([opoints, oH, oinvH, od0, od1, op0x, op0y], 'hbundle1')

    # STEP #4 Homography based calibration
    run_homography_based_calibration(opoints, oH, oinvH, od0, od1, op0x, op0y, IMAGE_WIDTH)

    # STEP #5 Metric Reconstruction & bundle adjustment
    # Still to implement


run_planar_calibration()