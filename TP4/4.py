import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import exposure

def Show(img):
    cv.imshow("Pic",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img1 = cv.imread("TP4/in/1.jpg",cv.IMREAD_GRAYSCALE)
Show(img1)

plt.hist(img1.ravel(),256,[0,256])

# Manual Thersholding 
Threshold = 210 
ret1, threshimg1 = cv.threshold(img1,Threshold,255,cv.THRESH_BINARY)
Show(threshimg1)
imageThresholded = 255 - threshimg1
Show(imageThresholded)

# Otsu Thresholding 
img1Preprocessed = exposure.adjust_gamma(img1,gamma=1.5,gain=1)
ret2, thresh2img1 = cv.threshold(img1Preprocessed,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
Show(thresh2img1)

img2 = cv.imread("TP4/in/3.jpg",cv.IMREAD_GRAYSCALE)
img2 = cv.resize(img2,(600,600))
ret3, threshimg2 = cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
Show(threshimg2)

th4 = cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,8)
Show(th4)

th5 = cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,8)
Show(th5)

thresh2img1Neg = 255 - thresh2img1 
Show(thresh2img1Neg)
thresh2img1Neg1 = cv.morphologyEx(thresh2img1Neg,cv.MORPH_CLOSE,np.ones((3,3),dtype=np.uint8))
Show(thresh2img1Neg1)
thresh2img1Neg2 = cv.morphologyEx(thresh2img1Neg1,cv.MORPH_OPEN,np.ones((3,3),dtype=np.uint8))
Show(thresh2img1Neg2)

C,H = cv.findContours(thresh2img1Neg2,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
img1_new = cv.drawContours(img1,C,-1,(0,255,0),1)
Show(img1_new)








