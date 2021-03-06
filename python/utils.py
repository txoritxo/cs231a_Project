import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
import os

#class distortion_model:
#    def __init__(self):


def qplot_correspondences2(img1, img2, good, mask):
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=mask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(img1.im, img1.kp, img2.im, img2.kp, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()
    cv.waitKey(0)



def plot_homographies(images):
    added_image = images[0].im
    for img in images[1:]:
        h, w = img.im.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, img.H)
        im_dst = cv.warpPerspective(img.im, img.H, dsize=(w, h))
        added_image = cv.addWeighted(added_image, 0.6, im_dst, 0.3, 0)
        added_image = cv.polylines(added_image, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        cv.imshow("Image1", added_image)
        cv.waitKey(0)
        z = 0


def qwarpPerspective(img, M, dsize):
    dst = np.zeros_like(img)
    dsize = img.shape
    coord = np.zeros((2, dsize[0]*dsize[1]))
    mask  = np.zeros((1, dsize[0]*dsize[1]))

    p0 = np.indices((dsize[1], dsize[0]))
    p0 = p0.reshape((2, dsize[1] * dsize[0]))
    p0 = np.vstack((p0, np.ones((1, p0.shape[1]))))

    res = np.dot(M, p0)
    res = res/[res[-1]]
    qimg = np.zeros_like(img)
    h, w = dsize
    for i in range(res.shape[1]):
        x, y, _ = res[:, i]
        if x >= 0 and x < w and y >= 0 and y < h:
            mask[0,i] = 1
            qimg[int(y), int(x)] = img[int(p0[1, i]), int(p0[0, i])]

    return qimg, res, mask


def qplot_correspondences(imgs):
    for img in imgs[1:]:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=img.mask,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(img.im, img.kp, imgs[0].im, imgs[0].kp, img.good, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()
        cv.waitKey(0)



def plot_homographies2(images):
    added_image = images[0].im
    for img in images[1:]:
        im_dst = cv.warpPerspective(img.im, img.H, dsize=(img.im.shape[1], img.im.shape[0]))
        im_dst2, coords, mask = qwarpPerspective(img.im, img.H, dsize=(img.im.shape[1], img.im.shape[0]))
        #im_dst2, coords, mask = qwarpPerspective(img.im, img.H, dsize=(1024, 768))
        added_image = cv.addWeighted(added_image, 0.2, im_dst2, 0.8, 0)
        #added_image = cv.polylines(added_image, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        cv.imshow("Image1", added_image)
        cv.waitKey(0)
    z = 0


def validate_correspondences(img0, img1, pt0, pt1):
    big_image = np.concatenate((img1.im, img0.im), axis=1)
    h, w = img0.im.shape
    plt.imshow(big_image, cmap='gray')
    pth0 = np.vstack((pt0, np.ones((1, pt0.shape[1]))))
    for i in range(pt0.shape[1]):
        x0, y0 = pt0[0,i]+w, pt0[1, i]
        x1, y1 = pt1[0, i], pt1[1, i]
        plt.plot([x0,x1],[y0,y1], '-ro', linewidth=0.2, markersize=1)

    pth1 = np.linalg.inv(img1.H) @ pth0
    pth1 = pth1/pth1[-1]
    for i in range(pt0.shape[1]):
        x1, y1 = pth1[0, i], pth1[1, i]
        plt.plot(x1, y1, 'bx', linewidth=0.2, markersize=5)
    plt.show()
    cv.waitKey(0)

def qtest_ba_parameter_extraction(imgs, y0, p, H, y_camera, camera_indices):
    for i, camera_index in enumerate(camera_indices[1:]):
        big_image = np.concatenate((imgs[camera_index].im, imgs[0].im), axis=1)
        h, w = imgs[0].im.shape
        plt.imshow(big_image, cmap='gray')
        for k, pt_index in enumerate(y_camera[camera_index]):
            y = y0[pt_index]
            pt1 = p[camera_index][k]
            _x0, _y0 = y[0] + w, y[1]
            _x1, _y1 = pt1[0], pt1[1]
            plt.plot([_x0, _x1], [_y0, _y1], '-ro', linewidth=0.2, markersize=1)
            z=0
        plt.show()
        cv.waitKey(0)

def reprojection_error(p, y_camera, H, camera_indices):
    reproj_error = [[] for i in range(len(y_camera))]  # number of images
    reproj_error_summary = np.zeros((len(camera_indices),4))
    for i in range(1,len(camera_indices)):   # camera with at least 10 matches
        phat_i = cv.perspectiveTransform(p[0][y_camera[i]].reshape(-1, 1, 2), H[i]).reshape(-1, 2)
        #phat2_0 = (np.column_stack((p[i], np.ones((len(p[i]), 1)))) @ H[i].T)
        #phat2a_0 = phat2_0[:,:2] / np.outer(phat2_0[:,2], np.ones(2))
        reproj_error[i] = np.linalg.norm(phat_i - p[i], axis=1)
        reproj_error_summary[i] = np.array([camera_indices[i], len(p[i]), np.sum(reproj_error[i]), np.mean(reproj_error[i])])
    reproj_error_total = np.sum(reproj_error_summary[:,2])
    return reproj_error_total, reproj_error_summary, reproj_error


def reprojection_inverse_error(p, y_camera, invH, camera_indices):
    reproj_inv_error = [[] for i in range(len(y_camera))]  # number of images
    reproj_inv_error_summary = np.zeros((len(camera_indices),5))
    for i in range(1,len(camera_indices)):   # camera with at least 10 matches
        s = np.linalg.svd(H[i], full_matrices=True, compute_uv=False)
        p_0 = p[0][y_camera[i]]
        phat_0 = cv.perspectiveTransform(p[i].reshape(-1, 1, 2), invH[i]).reshape(-1, 2)
        reproj_inv_error[i] = np.linalg.norm(phat_0 - p_0, axis=1)
        reproj_inv_error_summary[i] = np.array([camera_indices[i], len(p_0), np.sum(reproj_inv_error[i]),
                                                np.mean(reproj_inv_error[i]), s[0]/s[-1]])
    reproj_inv_error_total = np.sum(reproj_inv_error_summary[:,2])
    return reproj_inv_error_total, reproj_inv_error_summary, reproj_inv_error


def create_mosaic(imgs, max_pictures=None, width = 1024, ncols=5):
    n = len(imgs) if max_pictures is None else min(len(imgs), max_pictures)
    ncols = min(ncols, n)
    nrows = n // ncols
    h0, w0 = imgs[0].im.shape
    f = w0 / h0
    shrink = (width / ncols)/w0
    w = int(w0 * shrink)
    h = int(h0 * shrink)
    the_pic = np.zeros((h*nrows, w*ncols))
    for r in range(nrows):
        for c in range(ncols):
            id = r*ncols + c
            im = cv.resize(imgs[id].im, dsize=(w, h))
            the_pic[h*r:h*(r+1), w*c:w*(c+1)] = im
            the_pic[:, w * c] = 0
            the_pic[:, w * (c + 1)-1] = 0
        the_pic[h * r, :] = 0
        the_pic[h * (r+1)-1, :] = 0
    plt.imshow(the_pic, cmap='gray')
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

def imase_seq_to_video(dir):
    i = 0
    img_array = []
    for filename in os.listdir(dir):
        if filename.lower().endswith(".jpg"):
            print('processing {}'.format(filename))
            img = cv.imread(dir+filename)
            h, w, l = img.shape
            sz = (w, h)
            img_array.append(img)
        else:
            continue

    out = cv.VideoWriter(dir + 'qVideo1.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, sz)

    for im in img_array:
        out.write(im)
    out.release()


def to_file(data, filename):
    with open('{}.pkl'.format(filename), 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def from_file(filename):
    with open('{}.pkl'.format(filename), 'rb') as infile:
        result = pickle.load(infile)
        return result
    return None

