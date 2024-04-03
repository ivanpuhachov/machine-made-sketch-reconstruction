import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

for k, v in os.environ.items():
        if k.startswith("QT_") and "cv2" in v:
            del os.environ[k]


def make_boundary(cnts, background_image, dist_thr=0):
    """
    Iterates through contours, simplifies by distance threshold
    :param cnts:
    :param background_image: (HxW), 1 - background, 0 - shape
    :param dist_thr:
    :return:
    """
    # print(background_image.shape)
    bnd = np.zeros(shape=(0, 2), dtype=int)
    edges = list()
    # TODO: with cv.CHAIN_APPROX_NONE we interpret contour as one closed curve
    for c in cnts:
        continuous_boundary = True
        start_index = bnd.shape[0]
        for p in c:
            # p.shape == (1,2)
            # if background_image[p[0, 1],p[0, 0]] > 0.5:
            #     # if its a background point we drop it
            #     continue
            if len(bnd)==0:
                # add the first point anyway
                bnd = np.vstack((bnd, p))
                continuous_boundary = True
                continue
            mindist = np.min(np.sum((bnd - p)**2, axis=1))
            if mindist > dist_thr:
                bnd = np.vstack((bnd, p))
                if continuous_boundary:
                    edges.append([bnd.shape[0]-1, bnd.shape[0]-2])
                continuous_boundary = True
        if bnd.shape[0] - start_index > 2:
            edges.append([bnd.shape[0]-1, start_index])
    edges.append([bnd.shape[0] - 1, 0])
    return bnd, edges


def get_boundary(pngfile, dist_thr=0):
    src = cv.imread(pngfile)
    print(src.shape)
    print(np.max(src))
    im_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(im_gray, (3, 3))
    threshold = 100
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # print("im gray", im_gray.shape)
    # print(im_gray)
    background_image = (1 - np.array(im_gray) / 255).astype(int)
    # plt.imshow(background_image)
    # plt.colorbar()
    # plt.title("im background_image")
    # plt.show()
    return make_boundary(cnts=contours, background_image=background_image, dist_thr=dist_thr)


def get_boundary_numpy(img, dist_thr=0):
    """

    :param dist_thr:
    :param img: (400x400) np array
    :return:
    """
    # TODO: these manipulations are risky, redo carefully
    img = 255 * np.repeat(1 - img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    print("len contours ", len(contours))
    # print(contours)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    cntr = contours[0]
    # print(len(cntr))
    return make_boundary(cnts=cntr, background_image=img, dist_thr=dist_thr)


if __name__ == "__main__":
    data = np.load("data/00000136.npz")
    im = data['Background']
    print("image shape", im.shape)
    # plt.imshow(im)
    # plt.colorbar()
    # plt.show()

    bbb, eee = get_boundary_numpy(im, dist_thr=0)
    # bbb, eee = get_boundary("data/00000004_back.png")
    print("boundary points: ", bbb.shape)
    print("boundary edges: ", len(eee))
    # print(bbb)
    # print(eee)

    plt.figure()
    plt.imshow(im, cmap='gray_r')
    # plt.scatter(bbb[:,0], 400 - bbb[:,1])
    plt.scatter(bbb[:,0], bbb[:,1])
    plt.axis("equal")
    plt.show()

