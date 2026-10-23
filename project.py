import cv2 as cv
img =cv.imread('/home/dhruv/Desktop/Photos/road2.jpeg')
cv.namedWindow("road2", cv.WINDOW_GUI_NORMAL)
cv.resizeWindow("road2", 600, 600) 
cv.imshow('road2',img)
cv.waitKey(0)