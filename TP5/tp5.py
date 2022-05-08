import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def mc(img1, img2, border_size):
    x1, y1 = img1.shape 
    x2, y2 = img2.shape 

    if (x1, y1) != (x2, y2):
        print ("Images de tailles differentes !")

    ext1 = img1[border_size:x1-border_size, border_size:y1-border_size]
    ext2 = img2[border_size:x1-border_size, border_size:y1-border_size]
    r = np.mean((ext1 - ext2) ** 2)
    return r


def mediane(img, size): 
    valNear = []
    res = img.copy()
    i = 0
    j = 0
    while(i < len(img)):
        while(j < len(img)):
            if((i == 0) or (j == 0) or (i == len(img) - 1) or (j == len(img) - 1)):
                res[i][j] = img[i][j]
            else:
                m = i - size
                n = j - size
                while(m < (i + size)):
                    while(n < (j + size)):
                        if((m >= 0) and (n >= 0) and ((i + size <= len(img))) and ((j + size <= len(img)))):
                            valNear.append(img[m][n])
                        n += 1
                    m += 1
                    n = j - size
                res[i][j] = np.median(valNear)
                valNear = []
            j += 1
        i += 1
        j = 0
    return res



img = cv.imread('photophore.tif', cv.IMREAD_GRAYSCALE)
imgGauss = cv.imread('ph_gauss.tif', cv.IMREAD_GRAYSCALE)
imgPulse = cv.imread('ph_pulse.tif', cv.IMREAD_GRAYSCALE)

imgGaussMaskmediane = mediane(imgGauss, 2)
imgPulseMaskmediane = mediane(imgPulse, 2)

mask = np.ones((5,5)) / 25
imgGaussMask = cv.filter2D(imgGauss, -1, mask, borderType=cv.BORDER_CONSTANT)
imgPulseMask = cv.filter2D(imgPulse, -1, mask, borderType=cv.BORDER_CONSTANT)

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(imgPulse, cmap='gray')
axarr[1].imshow(imgPulseMaskmediane, cmap='gray')
plt.show()

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(imgGauss, cmap='gray')
axarr[1].imshow(imgGaussMaskmediane, cmap='gray')
plt.show()

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(imgPulse, cmap='gray')
axarr[1].imshow(imgPulseMask, cmap='gray')
plt.show()

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(imgGauss, cmap='gray')
axarr[1].imshow(imgGaussMask, cmap='gray')
plt.show()

varimgGaussMask = mc(img, imgGaussMask, 3)
varimgPulseMask = mc(img, imgPulseMask, 3)
varImgGaussMaskmediane = mc(img, imgGaussMaskmediane, 3)
varImgPulseMaskmediane = mc(img, imgPulseMaskmediane, 3)

print("Gaussian image with average mask => " + str(round(varimgGaussMask)))
print("Pulse image with average mask  => " + str(round(varimgPulseMask)))
print("Gaussian image with middle mask  => " + str(round(varImgGaussMaskmediane)))
print("Pulse image with middle mask => " + str(round(varImgPulseMaskmediane)))
