import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 

seuil = 1500
seuil_haut = 150
seuil_bas = 0.7 * seuil_haut
im = cv.imread('photophore.tif', cv.IMREAD_GRAYSCALE)
refCanny = cv.Canny(im, seuil_haut, seuil_bas)

def contour_laplacien(im_src, seuil):
    
    flt1 = np.zeros((3,3))
    flt1[0][1] = 1
    flt1[1][0] = 1
    flt1[1][1] = -4
    flt1[1][2] = 1
    flt1[2][1] = 1

    imlap = cv.filter2D(im, -1, flt1, borderType=cv.BORDER_CONSTANT)
    impol = calcul_polarite(imlap)

    impol0 = impol[:-1,1:]
    impol1 = impol[:-1,:-1]
    impol2 = impol[1:,:-1]
    
    imzero = np.logical_or(impol0 != impol1, impol0 != impol2).astype(int)

    x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y = x.transpose()
    gradient = cv.filter2D(im, -1, x, borderType=cv.BORDER_CONSTANT).astype(int) ** 2 + cv.filter2D(im, -1, y, borderType=cv.BORDER_CONSTANT).astype(int) ** 2
    gradient = gradient[:-1, :-1]
    return np.logical_and(imzero,(gradient > seuil))

reflapla = contour_laplacien(im, seuil)

def calcul_polarite(imlap):
    return (imlap > 0).astype(int)


for i in range(5,25,5):
    im1 = (im + i*np.random.randn(im.shape[0], im.shape[1])).astype('uint8')
    difCanny = refCanny != cv.Canny(im1, seuil_bas, seuil_haut)
    tauxCanny = difCanny.sum()
    print("Canny",i," => ", tauxCanny)

for i in range(5,25,5):
    im2 = (im + i*np.random.randn(im.shape[0], im.shape[1])).astype('uint8')
    ftl2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    contourLap = cv.filter2D(im2, -1, ftl2,borderType=cv.BORDER_CONSTANT)
    difLaplace = np.not_equal(reflapla, contourLap[:-1,:-1])
    tauxLaplace = difLaplace.sum()
    print("LaPlace",i," => ", tauxLaplace)

fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(im, cmap='gray')
axarr[1].imshow(refCanny, cmap='gray')
axarr[2].imshow(reflapla, cmap='gray')
plt.show()






