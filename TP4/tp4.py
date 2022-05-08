import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def variance_locale(img, mask):
    l, c = img.shape
    res = np.ones((l, c))
    m = np.floor(mask / 2.)

    m = int(m)

    for i in range(m, l-m):
        for j in range(m, c-m):
            a = img[i-m:i+m, j-m:j+m]
            v = a.var()
            res[i, j] = v

    return res 

msk_lap = np.array(np.mat('-1 0 2 0 -1; -4 0 8 0 -4; -6 0 12 0 -6; -4 0 8 0 -4; -1 0 2 0 -1'))

im1 = cv.imread('texture1.tif', cv.IMREAD_GRAYSCALE)
im2 = cv.imread('texture2.tif', cv.IMREAD_GRAYSCALE)
im3 = cv.imread('texture3.tif', cv.IMREAD_GRAYSCALE)

imgConvol1 = cv.filter2D(im1, -1, msk_lap, borderType=cv.BORDER_CONSTANT)
imgConvol2 = cv.filter2D(im2, -1, msk_lap, borderType=cv.BORDER_CONSTANT)
imgConvol3 = cv.filter2D(im3, -1, msk_lap, borderType=cv.BORDER_CONSTANT)

attImg1 = variance_locale(imgConvol1, 10)
attImg2 = variance_locale(imgConvol2, 10)
attImg3 = variance_locale(imgConvol3, 10)

aveImg1 = attImg1.mean()
aveImg2 = attImg2.mean()
aveImg2 = attImg3.mean()

ecartImg1 = np.sqrt(attImg1.var())
ecartImg2 = np.sqrt(attImg2.var())
ecartImg3 = np.sqrt(attImg3.var())

fig, axarr = plt.subplots(4)
axarr[0].hist(attImg1.ravel(), 100)
axarr[1].hist(attImg2.ravel(), 100)
axarr[2].hist(attImg3.ravel(), 100)
axarr[3].hist(im3.ravel(), 100)
plt.show()

imgThr3 = cv.threshold(im3, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

fig, axarr = plt.subplots()
axarr.imshow(imgThr3[1], cmap='gray')
plt.show()
