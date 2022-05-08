import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

image_size = 256




axA = plt.axes([0.2, 0.05, 0.6, 0.03]) # [gauche, bas, largeur, hauteur] 
axB = plt.axes([0.2, 0.08, 0.6, 0.03]) # en dimensions normalisées

s_A = Slider(axA, 'A', 0, image_size)
s_B = Slider(axB, 'B', 0, image_size)
# frequences respectivement dans les directions x et y
x, y = np.meshgrid(range(0, 256), range(0, 256))
A = s_A.val
B = s_B.val
i =128* (np.sin(2*np.pi*(A*x +B* y ))+1)