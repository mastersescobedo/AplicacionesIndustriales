# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
# https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection
# https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
# https://es.mathworks.com/help/images/texture-segmentation-using-gabor-filters.html
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

import numpy as np
from sklearn.cluster import KMeans
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


def build_filters():
    filters = []
    ksize = 51
    for theta in np.arange(0, np.pi, np.pi / 32):
        kern = cv2.getGaborKernel((ksize, ksize), sigma=9.0, theta=theta, lambd=20.0, gamma=1, psi=0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


if __name__ == '__main__':
    import sys

    print(__doc__)

    try:
        img_fn = sys.argv[1]
    except:
        img_fn = 'test.png'

    img = cv2.imread(img_fn)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)

    fs = cv2.FileStorage("/home/sergio/PycharmProjects/AplicacionesInd/Ejer3/Test/25.reg", cv2.FILE_STORAGE_READ)
    fn = fs.getNode("rectangles")
    print(fn.mat())
    aux = fn.mat()



    filters = build_filters()

    res1 = process(img, filters)

    cv2.imshow('result', res1)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()