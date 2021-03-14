import numpy as np

def build_K(f0, f1, p0x, p0y):
    return np.array([f0, 0, p0x, 0, f1, p0y, 0, 0, 1]).reshape(3, 3)

def pack_parameters(K, n):
    # K is the camera calibration matrix (3x3)
    # n is the normal to the scene in the frame associated to camera 0
    K_params = 4
    n_params = 3
    X = np.zeros(K_params + n_params)
    X[:K_params] = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    X[K_params:] = n
    return X

def unpack_parameters(X):
    K = np.zeros((3, 3))
    K[0, 0] = X[0]
    K[1, 1] = X[1]
    K[0, 2] = X[2]
    K[1, 2] = X[3]
    K[2, 2] = 1
    n = X[4:].reshape(3, 1)
    return K, n

def compute_a_b(n, e):
    nn = n.reshape(3, 1)
    ee = e.reshape(3, 1)

    b = np.cross(nn.T, ee.T).T
    a = np.cross(b.T, nn.T).T
    return a, b

def unit_norm_constraint(X):
    K, n0 = unpack_parameters(X)
    return 1-np.linalg.norm(n0)


def compute_residual(X, H, e):
    #e = np.array([1, 0, 0]).reshape(3, 1)
    K, n0 = unpack_parameters(X)
    #n0 = n0 / np.linalg.norm(n0)
    a0, b0 = compute_a_b(n0, e)
    invK = np.linalg.inv(K)
    temp = invK @ H @ K
    a = temp @ a0
    b = temp @ b0
    at = np.swapaxes(a, 1,2)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    r1 = (at @ b) / (na * nb)
    r2 = 1 - (nb**2 / na**2)
    res = np.abs(np.sum(r1) + np.sum(r2))
    #print('residual: {}'.format(res))
    return res


def compute_initial_f(H, Tscale):
    # we have to solve the following linear system on sq_f:
    # H_31 H_32 sq_f + H_11 H_12 + H_21 H_22 = 0
    # H_31^2 sq_f - H_32^1 sq_f+ H_11^2 + H_21^2 - H_12^2 - H_22^2 = 0
    # which leads to an overdetermined system AX + B = 0
    numH = len(H)-1
    B1 = H[1:, 0, 0] * H[1:, 0, 1] + H[1:, 1, 0] * H[1:, 1, 1]
    B2 = H[1:, 0, 0]**2 + H[1:, 1, 0]**2 - H[1:, 0, 1]**2 - H[1:, 1, 1]**2
    A1 = H[1:, 2, 0] * H[1:, 2, 1]
    A2 = H[1:, 2, 0]**2 - H[1:, 2, 1]**2
    A = np.hstack((A1, A2)).reshape((numH*2), 1)
    B = -np.hstack((B1, B2)).reshape((numH*2), 1)
    x, res, rank, sval = np.linalg.lstsq(A, B, rcond=None)
    # warning, x could be negative!!! which means that f=sqrt(x) would be complex
    # This might be wrong, but we're returning here f=sqrt(abs(x)) !!
    # the result make sense, but again, it might be wrong
    if x < 0:
        print('warning initial estimation of f^2 is <0!, taking its absolute value instead!')
    print('initial f is :{}'.format(np.sqrt(np.abs(x)) / Tscale[0,0]))
    return np.asscalar(np.sqrt(np.abs(x)))

