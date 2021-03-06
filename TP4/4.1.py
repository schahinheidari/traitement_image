import cv2 as cv
import numpy as np 

def Show(img):
    cv.imshow("Pic",img)
    cv.waitKey()
    cv.destroyAllWindows()
    
img1 = cv.imread("TP4/in/sodoko.jpg",cv.IMREAD_UNCHANGED)
img1 = cv.resize(img1,(600,600))
Show(img1)

gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150)
Show(edges)

lines = cv.HoughLines(edges,1,np.pi/180,200)

for x in range(np.shape(lines)[0]):
    L = lines[x,:,:]
    rho = L[:,0]
    theta = L[:,1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 +1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img1,(x1,y1),(x2,y2),(255,0,0),3)
Show(img1)   

lines1 = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

for line in lines1:
    x1,y1,x2,y2 = line[0]
    cv.line(img1,(x1,y1),(x2,y2),(255,0,255),3)
Show(img1)   

coins = cv.imread("TP4/in/2.jpg",cv.IMREAD_COLOR)
coins = cv.resize(coins,(600,600))
Show(coins)   
grayCoins = cv.cvtColor(coins,cv.COLOR_BGR2GRAY)
circles = cv.HoughCircles(grayCoins,cv.HOUGH_GRADIENT,1.1,10,90)
circles = np.round(circles)

for i in circles[0,:]:
    center = (i[0],i[1])
    radius = i[2]
    cv.circle(coins,center,radius,(0,0,255),3)
    
Show(coins)