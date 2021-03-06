import numpy as np
from utils import *
from homography_based_calibration import *
from scipy.optimize import NonlinearConstraint
from scipy import optimize

def run_homography_based_calibration(points, H, invH, d0, d1, p0x, p0y, imw=640):
    print('STEP #4 - Conducting Homography based calibration')
    f0 = compute_initial_f(invH)
    n0 = np.array([0, 0, 1])
    K0 = build_K(f0, f0, p0x, p0y)
    X0 = pack_parameters(K0, n0)
    e = np.array([1, 0, 0]).reshape(3, 1)
    K, n = unpack_parameters(X0)
    res0 = compute_residual(X0, invH, e)
    nlc = NonlinearConstraint(unit_norm_constraint, -0.001, 0.001)
    res = optimize.minimize(compute_residual, X0, method='slsqp', jac='3-point', tol=1e-15,
             options={'verbose': 1}, constraints=nlc, args = (invH, e) )
    K, n = unpack_parameters(res.x)
    a, b = compute_a_b(n, e)
    print('Finished optimization: output message: {}'.format(res.message))
    print('success: {} ; iters: {} ; residual: {}'.format(res.success, res.nit, res.fun))
    print('K =\n {} ; \nn = {}'.format(K, n.T))
    print('a =\n {} ; b = {}'.format(a.T, b.T))
    img_sz = np.array([imw, imw/1.5])
    sensor_size = np.array([23.5, 15.6])
    factor = img_sz / sensor_size
    focal0 = K[0, 0] / factor[0]
    focal1 = K[1, 1] / factor[1]
    print('metric focal distances are: [ {}, {} ]'.format(focal0, focal1))
    return K, n, a, b



def run_calibration_from_file():
    points, H, invH, d0, d1, p0x, p0y = from_file('hbundle1')
    run_homography_based_calibration(points, H, invH, d0, d1, p0x, p0y)


#run_calibration_from_file()
