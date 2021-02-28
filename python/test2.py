import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.optimize as opt
import math
import os
#from mpl_toolkits.mplot3d import Axes3D

# Calculates Rotation Matrix given euler angles.
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def qtest1():
    Rx = 45
    Ry = 60
    Rz = -25
    rot_angles = np.array([math.radians(Rx), math.radians(Ry), math.radians(Rz)])
    R = eulerAnglesToRotationMatrix(rot_angles)
    T = np.array([10, -5, 15])
    X = np.array([33, -25, 45])

    X_hat = R @ X + T
    X_back = R.T @ X_hat - R.T @ T
    print('X = {}'.format(X))
    print('X_hat = {}'.format(X_hat))
    print('X_back = {}'.format(X_back))
    z=0

def resize_file(dir, filename, width, suffix=''):
    im = cv.imread(dir+'/'+filename)
    h, w, _ = im.shape
    if h > w:
        im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
        h, w, _ = im.shape
    f = w / h
    new_height = int(width/f)
    im = cv.resize(im, dsize=(width, new_height))
    newname = '{}/resized/RESIZED_{}.jpg'.format(dir, suffix)
    cv.imwrite(newname, im)
    z=0


def resize_directory(dir):
    i=0
    if not os.path.exists(dir+'/resized/'):
        os.makedirs(dir+'/resized/')

    for filename in os.listdir(dir):
        if filename.lower().endswith(".jpg") or filename.endswith(".png"):
            print('processing {}'.format(filename))
            suffix = '{:002d}'.format(i)
            resize_file(dir, filename, width=800, suffix=suffix)
            i+=1
        else:
            continue

def fun(x):
    return [x[0] + 0.5 * x[1] - 1.0,
            0.5 * (x[1] - x[0]) ** 2]


def qplot3d(H):
    X0 = np.array([100, 100, 50, 1]).reshape(4, 1)
    X1 = np.array([100, 200, 50, 1]).reshape(4, 1)
    X2 = np.array([200, 200, 50, 1]).reshape(4, 1)
    X3 = np.array([200, 100, 50, 1]).reshape(4, 1)
    X4 = X0

    X = np.hstack((X0, X1, X2, X3, X4))
    XX = H @ X
    XX = XX/XX[-1,:]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(X[0, :], X[1, :], X[2, :], 'gray')
    ax.plot3D(XX[0, :], XX[1, :], XX[2, :], 'green')
    set_axes_equal(ax)
    plt.show()
    cv.waitKey(0)

def test_matrix(H):
    X0 = np.array([100, 100, 0, 1]).reshape(4, 1)
    X1 = np.array([100, 200, 0, 1]).reshape(4, 1)
    X2 = np.array([200, 200, 0, 1]).reshape(4, 1)
    X3 = np.array([200, 100, 0, 1]).reshape(4, 1)
    X4 = X0

    X = np.hstack((X0, X1, X2, X3, X4))
    XX = H @ X
    XX = XX/XX[-1, :]

    M = np.identity(3)
    M = np.hstack((M, np.zeros((3, 1))))
    M[-1, -1] = 1
    p0 = M @ X
    p0 = p0/p0[-1]
    p = M @ XX
    p = p/p[-1]

    plt.plot(p[0, :], p[1, :], '-ro', linewidth=1)
   # plt.plot(p0[0, :], p0[1, :],  '-gx', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
#    for i in range(pt0.shape[1]):
#        x0, y0 = pt0[0,i]+w, pt0[1, i]
#        x1, y1 = pt1[0, i], pt1[1, i]
#        plt.plot([x0,x1],[y0,y1], '-ro', linewidth=0.2, markersize=1)
    plt.show()
    cv.waitKey(0)

def qrename_files(dir):
    i = 0
    for filename in os.listdir(dir):
        if filename.lower().endswith(".jpg"):
            print('processing {}'.format(filename))
            suffix = '{:04d}'.format(i)
            oldname = '{}/{}'.format(dir, filename)
            newname = '{}/DSCF{}.jpg'.format(dir, suffix)
            os.rename(oldname, newname)
            i+=1
        else:
            continue



qrename_files('../data/quere/x-e3_18mm/photosPicture1/')
#resize_directory(r'C:\Users\Carlos\development\Stanford\ComputerVision\project\data\checkerboard0\subset3000')
#qtest1()