import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt
from numpy.random import randn
import math

im = cv.imread('photophore.tif', cv.IMREAD_GRAYSCALE)
#im = cv.imread('photophore.tif', 0)

et = 10
imb = im + et*randn(im.shape[0], im.shape[1])

# moyen filter
flt1 = np.ones((3,3)) / 9
flt2 = np.ones((5,5)) / 25

#By using the convolution operator of the OpenCV module
imf1 = cv.filter2D(imb, -1, flt1, borderType=cv.BORDER_CONSTANT)
imf2 = cv.filter2D(imb, -1, flt2, borderType=cv.BORDER_CONSTANT)


#gaussien filter
X, Y = np.meshgrid(range(-1,2), range(-1,2))

X = np.array([[-1,0,1],
             [-1,0,1],
             [-1,0,1]])

Y = np.array([[-1,-1,-1],
             [-1,-1,-1],
             [-1,-1,-1]])
'''
def sigma(first, last, const):
    sum = 0
    for i in range(first, last + 1):
        sum += const * i
    return sum
'''

sigma = 9
#construction of normalisation filter
fgauss = np.exp(-(X*X + Y*Y) / (2* sigma* sigma))
fgauss = fgauss / sum(sum(fgauss))

# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 1
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(fgauss)
plt.axis('off')
plt.title("affichage première image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(fgauss)
plt.axis('off')
plt.title("affichage deuxième image")




plt.imshow(fgauss, cmap='gray')

plt.show()
