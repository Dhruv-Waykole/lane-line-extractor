import cv2 as cv
import numpy as np
img =cv.imread('/home/dhruv/Desktop/Photos/road1.jpg')


def rescaleimage(frame, scale=1):
    width=int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


blur=cv.GaussianBlur(img, (3,3),cv.BORDER_DEFAULT)
canny = cv.Canny(blur,125,175)
canny1 = cv.Canny(img,125,175)
resized= cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)   


white_mask = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 40, 255]))

yellow_mask = cv.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))

preview = img.copy()
def region_of_interest(image):
    h, w = image.shape[:2]
    polygon = np.array([[
    (int(0.00 * w), h),                 
    (int(0.35 * w), int(0.50 * h)),     
    (int(0.65 * w), int(0.50 * h)),     
    (int(1.00 * w), h)                 
 ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygon, 255)
    
    cv.polylines(preview, polygon, True, (0, 0, 255), 3)
    return cv.bitwise_and(image, mask)

lane_mask = cv.bitwise_or(white_mask, yellow_mask)
roi_mask = region_of_interest(lane_mask)
result2 = cv.bitwise_and(img, img, mask=roi_mask)

result = cv.bitwise_and(img, img, mask=lane_mask)


cv.imshow("Canny edges", canny )
cv.imshow("Canny edges1", canny1)
cv.imshow("blur", blur)
cv.imshow("img", rescaleimage(img))
cv.imshow("resized", resized)
cv.imshow("hsv", result)
cv.imshow("roi", result2)

cv.imshow("polygon preview", preview)
cv.waitKey(0)

