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
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta, lambd=10.0, gamma=1, psi=0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def contains_maloooooooo(R1, R2):
    #R1 contiene a R2
    out = 0
    if ((R2.x+R2.w) < ((R1.x-15)+(R1.w+30)) and (R2.x) > ((R1.x-15)) and (R2.y) > ((R1.y-15)) and (R2.y+R2.h) < (R1.y-15+R1.h+30)):
        return True
    else:
        return False

def contains(R1, R2, Correctos, FalsosPositivos):
    #R1 contiene a R2
    correcto_local=0
    FP_local=0
    for i in range(0,R1.shape[1]):
        if ((R2[0]+R2[2]) < ((R1[0][i]-15)+(R1[2][i]+30)) and (R2[0]) > ((R1[0][i]-15)) and (R2[1]) > ((R1[1][i]-15)) and (R2[1]+R2[3]) < (R1[1][i]-15+R1[3][i]+30)):
            correcto_local=1

    if correcto_local==0:
        FP_local= 1

    Correctos+=+correcto_local
    FalsosPositivos+=FP_local
    return Correctos, FalsosPositivos



if __name__ == '__main__':
    import sys

    print(__doc__)

    try:
        #img_fn = sys.argv[1]
        img_fn = "/home/sergio/PycharmProjects/AplicacionesInd/Ejer3/Train/25.png"
    except:
        img_fn = 'test.png'



    FalsosPositivos=0
    FalsosNegativos=0
    Correctos=0

    img = cv2.imread(img_fn)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)



    imgfinal = np.copy(img)
    img_ground = np.copy(img)


    fs =cv2.FileStorage("/home/sergio/PycharmProjects/AplicacionesInd/Ejer3/Train/25.reg",cv2.FILE_STORAGE_READ)
    fn=fs.getNode("rectangles")
    aux = fn.mat()


    for i in range(0,aux.shape[1]):
        cv2.rectangle(img_ground, (aux[0][i], aux[1][i]), (aux[0][i] + aux[2][i], aux[1][i] + aux[3][i]),(255,255,255),2)


    filters = build_filters()

    res1 = process(img, filters)
    (thresh, im_bw) = cv2.threshold(res1, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #im_bw = cv2.adaptiveThreshold(res1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)

    kernel_filter = np.ones((5, 5), np.uint8)
    kernel_fix = np.ones((3,3),np.uint8)
    erosion = cv2.erode(im_bw, kernel_filter, iterations=2)
    dilation = cv2.dilate(erosion, kernel_filter, iterations=2)
    ero2 = cv2.erode(dilation, kernel_fix, iterations=1)




    cv2.imshow('result', res1)
    cv2.waitKey(4000)

    # Busqueda de contornos y dibujarlos en la imagen aquellos que correspondan a nodos
    im2, contours, hierarchy = cv2.findContours(ero2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for z in range(len(contours)):
        con = contours[z]
        x, y, w, h = cv2.boundingRect(con)

        if (12 < w < 100 and 12 < h < 80) or (20 < w < 400 and 4 < h < 50):
            points = [x, y, w, h]
            cv2.rectangle(imgfinal, (x, y), (x + w, y + h), (255, 255, 255), 2)
            Correctos, FalsosPositivos = contains(aux, points, Correctos, FalsosPositivos)

    # Mostrar la imagen resultante
    plt.imshow(imgfinal, cmap=None), plt.title('title'), plt.show()
    plt.imshow(img_ground, cmap=None), plt.title('title'), plt.show()

    plt.imshow(im_bw, cmap=None), plt.title('title'), plt.show()
    #plt.imshow(erosion, cmap=None), plt.title('title'), plt.show()
    #plt.imshow(dilation, cmap=None), plt.title('title'), plt.show()
    plt.imshow(ero2, cmap=None), plt.title('title'), plt.show()


    print("correctos",Correctos)
    print("Falsos positivos",FalsosPositivos)

    cv2.destroyAllWindows()