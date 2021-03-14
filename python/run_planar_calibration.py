import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.optimize as opt
from collections import defaultdict
#from scipy.optimize import newton_krylov
from utils import *
from run_bundle_adjustment import *
from bundle_adjustment import *
from metric_reconstruction import *
from run_homography_based_calibration import *
#import run_homography_based_calibration2 as rhc
import metric_bundle_adjustment as mba

IMAGE_WIDTH = 800
LV_SCALE_PTS = False
LV_USE_FLANN_MATCH = True
LV_RESIZE = False

class qimage:
    def __init__(self, filename):
        self.im = cv.imread(filename, 0)
        h, w = self.im.shape

        if h>w:
            self.im = cv.rotate(self.im, cv.ROTATE_90_CLOCKWISE)
            h, w = self.im.shape

        if LV_RESIZE is True:
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
        self.invH = []
        self.maskinv = None
        self.good = None
        self.p0 = None
        self.p1 = None
        self.Tscale = np.diag(np.array([2./w, 2./w])) if LV_SCALE_PTS is True else np.eye(2)
        h, w = self.im.shape
        self.cen = np.array([1, h/w]) if LV_SCALE_PTS is True else np.array([w/2, h/2])
        self.scaled_dim = np.array([2*h/w, 2]) if LV_SCALE_PTS is True else np.array([h, w])

def compute_homography(img1:qimage, img2:qimage):
    # img1 = image i, img2 = image0
    MIN_MATCH_COUNT = 10
    MAX_MATCH_COUNT = 1000
    FLANN_INDEX_KDTREE = 1

    if LV_USE_FLANN_MATCH is True:
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        #search_params = dict(checks=50)
        search_params = dict(checks=100)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(img1.des, img2.des, k=2)
    else:
        bf = cv.BFMatcher()
        matches = bf.knnMatch(img1.des, img2.des, k=2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)

    # store all the good matches as per Lowe's ratio test.
    good = []
    num_matches = 0
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            num_matches += 1
        if num_matches >= MAX_MATCH_COUNT:
            break

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([img1.kp[m.queryIdx].pt for m in good]) @ img1.Tscale
        dst_pts = np.float32([img2.kp[m.trainIdx].pt for m in good]) @ img2.Tscale
        M, mask = cv.findHomography(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv.RANSAC, 5.0)
        Minv, maskinv = cv.findHomography(dst_pts.reshape(-1, 1, 2), src_pts.reshape(-1, 1, 2), cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        img1.good = good.copy()
        img1.mask = matchesMask.copy()
        img1.H = M
        img1.invH = Minv
        img1.maskinv = maskinv.ravel().tolist().copy()
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
    p0 = img0.Tscale @ p0
    p1 = img1.Tscale @ p1
    return p0, p1


def qload_images(idx0, lastidx):
    imgs = list()
    for i in range(idx0, lastidx):
        filename = '../data/quere/eos450/photosPicture1/DSCF{:04d}.jpg'.format(i)
        #filename = '../data/quere/x-e3_18mm/photosPicture1/DSCF{:04d}.jpg'.format(i)
        print('loading image {}'.format(filename))
        imgs.append(qimage(filename))
    print('STEP #1 - Loading images and identifying feature points :COMPLETED')
    return imgs


def qload_images_michael(idx_list):
    imgs = list()
    for idx in idx_list:
        filename = '../data/globeviewer/images/Rozas_gyro_bias_laps_007{:02d}_rgb.jpg'.format(idx)
        print('loading image {}'.format(filename))
        imgs.append(qimage(filename))
    print('STEP #1 - Loading images and identifying feature points :COMPLETED')
    return imgs


def qload_images_michael2():
    imgs = list()
    datadir = '../data/michael/Rozas_julio/'
    filelist = os.listdir(datadir)
    for filename in filelist:
        file = datadir + filename
        print('loading image {}'.format(file))
        imgs.append(qimage(file))
    print('STEP #1 - Loading images and identifying feature points :COMPLETED')
    return imgs


def qcompute_homographies(imgs):
    imgs[0].H = np.identity(3)
    imgs[0].invH = np.identity(3)
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
    # dict is a dictionary of observations in the reference image and number of images where a correspondence is found
    dict = defaultdict(int)
    for im in imgs[1:]:
        for match, mask in zip(im.good, im.mask):
            if not mask: continue
            dict[match.trainIdx] +=1

    #dict2 is the same dictionary as dict, but keeping only when correspondences are greater than min_matches
    dict2 = {key: val for key, val in dict.items() if val > min_matches}
    dict3 = {key: [0] for key in dict2}
    #y_camera contains a list of indices for each camera pointing to the corresponding observations in the reference image
    y_camera = [[] for i in range(len(imgs))]
    y0 = None # y0 list of observations in the reference image
    p = [None]*len(imgs) # list of observations in each of the images. p[i] contains a list of such observations in image i

    for pt_index, id in enumerate(dict2): #pt_index increasing, id is the id of the point in the ref image
        # p[i] is the container of observations for image 1. Image0 is the reference and shall contain
        # all the points in dict2
        # pt_index is the index of point id in container p[0].
        if p[0] is None:
            p[0] = np.array(imgs[0].Tscale @ imgs[0].kp[id].pt)
        else:
            p[0] = np.vstack((p[0], imgs[0].Tscale @ np.array(imgs[0].kp[id].pt)))
        y_camera[0].append(pt_index)
        #now we have to iterate in all other images and check whether a match exists with point id
        for i, im in enumerate(imgs[1:], start=1):
            for match, mask in zip(im.good, im.mask):
                if not mask: continue
                if match.trainIdx == id and pt_index not in y_camera[i]:
                    dict3[id].append(i) #dict3 contains lists of images that have correspondence with feature point id
                    y_camera[i].append(pt_index) #append pt_index to the list of point indices for camera i
                    if p[i] is None:
                        p[i] = np.array(im.Tscale @ im.kp[match.queryIdx].pt)
                    else:
                        p[i] = np.vstack((p[i], im.Tscale @ im.kp[match.queryIdx].pt))

    camera_indices = list(i for i, ob in enumerate(p) if len(ob) > 10)
    y0 = p[0].copy()
    H = np.zeros((len(camera_indices), 3, 3))
    invH = np.zeros((len(camera_indices), 3, 3))
    # condnum = np.zeros(len(camera_indices))
    # condnuminv = np.zeros(len(camera_indices))
    for i, cam_id in enumerate(camera_indices):
        temp = imgs[cam_id].H / imgs[cam_id].H[-1, -1]
        H[i] = temp
        temp = imgs[cam_id].invH/imgs[cam_id].invH[-1, -1]
        invH[i] = temp
        # s = np.linalg.svd(H[i], full_matrices=True, compute_uv=False)
        # condnum[i] = s[0]/s[-1]
        # s = np.linalg.svd(invH[i], full_matrices=True, compute_uv=False)
        # condnuminv[i] = s[0] / s[-1]
    # camera indices: includes indices of cameras with more than 10 observations
    # p. It is a list of numpy arrays. The ith element of the list contains the ordered observations in camera i
    # y_camera. it is a list of list of indices. The ith element of the list contains indices of the 3d points
    # which have correspondence observations in camera i
    # y0 is a numpy array containing the positions of the points in world coordinates, in this case, for initialization
    # purposes it is the same as the observations from camera 0
    # H is a numpy array of size [ncameras, 3, 3] which contains the stacked homography matrices
    return y0, p, invH, y_camera, camera_indices


def run_planar_calibration():
    #1 Set suffixes of pictures to load
    LV_IMAGES_QUERE = False
    # STEP #1. Feature Matching
    if LV_IMAGES_QUERE is True:
        idx0 = 0
        # lastidx = 27
        lastidx = 15
        # lastidx = 35
        imgs = qload_images(idx0, lastidx) # load images and compute features
    else:
        LV_GLOBEVIEWER = False
        if LV_GLOBEVIEWER is True:
            dist_ini = np.zeros(2)
            #idx_list = np.array([30, 44, 18, 61, 51, 45, 17, 27, 15, 25, 59, 33, 73,  4,  5, 52,  0, 10, 60,  7, 16])
            idx_list = np.array([30, 44, 18, 61, 51, 45, 17, 27, 15, 25])
            imgs = qload_images_michael(idx_list) # load images and compute features
        else:
            dist_ini = np.array([1e-4, 1e-6])
            imgs = qload_images_michael2() # load images and compute features

    create_mosaic(imgs, max_pictures=20)
    Tscale = np.eye(3)
    Tscale[:2,:2] = imgs[0].Tscale

    # STEP #2. Projective Reconstruction
    imgs = qcompute_homographies(imgs) # compute homographies with respect to image[0]
    # filter matches. We don't want to keep points that are only seen by a few cameras
    # the minimum matches considered are given by parameter min_matches
    #y0, p, invH, y_camera, camera_indices = qfilter_matches(imgs, min_matches=7)
    y0, p, invH, y_camera, camera_indices = qfilter_matches(imgs, min_matches=3)
    # this function is for visual validation purposes, it'll draw point correspondences for all considered images in
    # camera_indices. such point correspondences will be extracted from the output of qfilter_matches
    # qtest_ba_parameter_extraction(imgs, y0, p, H, y_camera, camera_indices)
    # plot_homographies(imgs)
    # qplot_correspondences(imgs)
    # qprojective_ba(imgs, the_camera)

    # set initial camera center and distortion
    #p0y, p0x = imgs[0].im.shape[0]/2, imgs[0].im.shape[1]/2
    #d0, d1 = 0.1, 0.001
    #X0, pdef0 = pack_parameters(y0, H, d0, d1, p0x, p0y)
    #res0 = compute_residual(X0, pdef0, p, y_camera, camera_indices, total_observations)
    #reprojT0, reprojS0, reprojE0 = reprojection_error(p, y_camera, invH, camera_indices)
    #reprojiT0, reprojiS0, reprojiE0 = reprojection_inverse_error(p, y_camera, H, camera_indices)

    # STEP #3 Projective Bundle Adjustment (4.2)
    dist_ini = np.zeros(2)
    opoints, oH, oinvH, od0, od1, op0x, op0y = run_bundle_adjustment(y0, p, invH, y_camera, camera_indices, imgs, dist_ini)
    to_file([opoints, oH, oinvH, od0, od1, op0x, op0y, y0, p, invH, y_camera, camera_indices], 'hbundle_adj_small')

    #reprojT1, reprojS1, reprojE1 = reprojection_error(p, y_camera, oinvH, camera_indices)
    #reprojiT1, reprojiS1, reprojiE1 = reprojection_inverse_error(p, y_camera, H, camera_indices)

    # X1, pdef1 = pack_parameters(opoints, oH, od0, od1, op0x, op0y)
    # res1 = compute_residual(X1, pdef1, p, y_camera, camera_indices, total_observations)
    # plot_pixel_errors(res0, res1)

    # STEP #4 Homography based calibration
    K, n, a, b = run_homography_based_calibration(opoints, oH, oinvH, od0, od1, op0x, op0y, IMAGE_WIDTH, Tscale)
    to_file([opoints, oH, oinvH, od0, od1, op0x, op0y, K, n, a, b, y0, p, invH, y_camera, camera_indices], 'homocalib_small')

    # STEP #5 Metric Reconstruction & bundle adjustment
    p3Dref, Rmats, Tvecs = metric_reconstruction(opoints, p, y_camera, camera_indices, K, a, b, n, od0, od1)
    d = np.array([od0, od1])
    X, pd = mba.pack_parameters(K, d, Rmats, Tvecs, p3Dref)
    oK, od, oRmats, oTvecs, op3Dref = mba.unpack_parameters(X, pd)
    total_observations = sum(len(p[c]) for c in camera_indices)

    res = least_squares(mba.compute_residual, X, verbose=2, x_scale='jac', method='trf', ftol=1e-6, loss='linear',
                        # loss='linear', #loss='cauchy', jac='2-point',#jac=mba.compute_mba_Jacobian
                        jac=mba.compute_mba_Jacobian, args=(pd, p, y_camera, camera_indices, total_observations))
    to_file((res,pd), 'full_ba_small')
    K_final, dist_final, fRmats, fTvecs, fp3Dref = mba.unpack_parameters(res.x, pd)

    print('K final = {}'.format(np.linalg.inv(Tscale) @ K_final))
    print('dist final = {}'.format(dist_final))
    qz = res.fun.reshape(2, -1)
    rep_error = np.std(qz, axis=1)
    print('reprojection error = {} norm= {}'.format(rep_error, np.linalg.norm(rep_error)))


run_planar_calibration()
