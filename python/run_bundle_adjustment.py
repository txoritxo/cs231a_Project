import numpy as np
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt
import cv2 as cv
from utils import *
from bundle_adjustment import *
from bundle_adjustment_utils import *


def run_bundle_adjustment(y0, p, H, y_camera, camera_indices, images):
    print('STEP #3 - Computing Projective Bundle Adjustment')
    total_observations = sum(len(p[c]) for c in camera_indices )
    p0y, p0x = images[0].im.shape[0] / 2, images[0].im.shape[1] / 2
    X, pdef = pack_parameters(y0, H, 0.01, 0.001, p0x, p0y)
    #plot_packed_homographies2(X, pdef, p, y_camera, camera_indices, total_observations, images)
    #plot_packed_correspondences(X, pdef, p, y_camera, camera_indices, total_observations, images)
    res0 = compute_residual(X, pdef, p, y_camera, camera_indices, total_observations)
    #plt.plot(res0)
    #plt.show()
    #cv.waitKey(0)
    t0 = time.time()
    res = least_squares(compute_residual, X, verbose=2, x_scale='jac', method='trf', ftol=1e-4, loss='linear', #loss='cauchy',
                        jac='3-point', args=(pdef, p, y_camera, camera_indices, total_observations))
    t1 = time.time()
    res1 = res.fun
    plot_pixel_errors(res0, res1)
    #plt.plot(res.fun)
    #plot_packed_homographies(X, pdef, p, y_camera, camera_indices, total_observations, images)
    opoints, oH, oinvH, od0, od1, op0x, op0y = unpack_parameters(res.x, pdef)
    print('p0 = [{:5.2f}, {:5.2f}]'.format(op0x, op0y))
    print('d = [{:5.2e}, {:5.2e}]'.format(od0, od1))
    #plot_adjusted_correspondences(res.x, pdef, p, y_camera, camera_indices, total_observations, images)
    #plot_packed_correspondences(res.x, pdef, p, y_camera, camera_indices, total_observations, images)
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    return opoints, oH, oinvH, od0, od1, op0x, op0y
