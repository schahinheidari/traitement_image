import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.measure import label, regionprops, regionprops_table

def Show(img):
    cv.imshow("Pic",img)
    cv.waitKey()
    cv.destroyAllWindows()
    
def Remove_objects(labelimg, threshold): 
    L1 = labelimg.copy()
    regions = regionprops(L1)
    for R in regions: 
        if R.area < threshold: 
            Label = R.label
            [r,c] = np.where(L1==Label)
            L1[r,c] = 0 
    return L1 

img1 = cv.imread("TP4/in/4.jpg", cv.IMREAD_UNCHANGED)
img1 = cv.resize(img1,(600,600))
Show(img1)

r = cv.selectROI(img1)
cv.destroyAllWindows()

filter = np.zeros(img1.shape[:2], dtype=np.uint8)
filter[r[1]:r[1]+r[3],r[0]:r[0]+r[2]] = 255
Show(filter)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

SegResults = cv.grabCut(img1, filter, r, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
InSegMap = SegResults[0]
InSegMapValues = set(list(InSegMap.ravel()))

FinalFilter = np.where((InSegMap==3)|(InSegMap==1), 255 ,0).astype('uint8')
Show(FinalFilter)

C,H = cv.findContours(FinalFilter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
newimg = cv.drawContours(img1, C, -1, (0,255,0), 2)
Show(newimg)

labelimg = label(FinalFilter)
print(set(list(labelimg.ravel())))
plt.Show(labelimg)
plt.show()

regions = regionprops(labelimg)
print(len(regions))

for P in regions: 
    print(P.perimeter)
    print(P.area)
    print(P.major_axis_length)
    print(P.minor_axis_length)
    print("---------------------------")
    
props = regionprops_table(labelimg, properties=('perimeter',
                                                 'area',
                                                 'major_axis_length', 
                                                 'minor_axis_length'))

print(props['area'][1])
NewLabelimg = Remove_objects(labelimg, 600)
plt.Show(NewLabelimg)
plt.show()

