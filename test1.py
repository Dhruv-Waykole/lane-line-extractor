import cv2 as cv
import numpy as np
img =cv.imread('/home/dhruv/Desktop/Photos/road2.jpg')


def region_of_interest(image, polygon):
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(image, mask)

def auto_polygon(img):
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(cv.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges[:int(0.4 * h), :] = 0          # ignore the sky area

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50,
                           minLineLength=40, maxLineGap=100)
    if lines is None:
        return None

    lines = lines.reshape(-1, 4)          # always (N, 4), works on any version

    left, right = [], []
    for x1, y1, x2, y2 in lines:          # no [:, 0] here
        if x1 == x2:
            continue
        m = (y2 - y1) / (x2 - x1)
        if abs(m) < 0.3 or abs(m) > 3:
            continue
        (left if m < 0 else right).append((m, y1 - m * x1))
    if not left or not right:
        return None

    ml, bl = np.median(left, axis=0)
    mr, br = np.median(right, axis=0)

    xv = (br - bl) / (ml - mr)           # vanishing point
    yv = ml * xv + bl
    y_top = int(yv + 0.1 * (h - yv))     # stop a bit below the horizon

    x_at = lambda m, b, y: int((y - b) / m)
    return np.array([[
        (x_at(ml, bl, h), h),
        (x_at(ml, bl, y_top), y_top),
        (x_at(mr, br, y_top), y_top),
        (x_at(mr, br, h), h),
    ]], dtype=np.int32)

polygon = auto_polygon(img)
if polygon is None:                      
    h, w = img.shape[:2]
    polygon = np.array([[(0, h), (int(.35*w), int(.5*h)),
                         (int(.65*w), int(.5*h)), (w, h)]], dtype=np.int32)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)   

white_mask = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 40, 255]))

yellow_mask = cv.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))    

lane_mask = cv.bitwise_or(white_mask, yellow_mask)


roi_mask = region_of_interest(lane_mask, polygon)
result = cv.bitwise_and(img, img, mask=roi_mask)
result2 = cv.bitwise_and(img, img, mask=lane_mask)

outline = img.copy()

cv.polylines(outline, polygon, True, (0, 0, 255), 3)

cv.imshow("polygon outline", outline)
cv.imshow("roi result", result)
cv.imshow("hsv", result2)

cv.imwrite("roi.png", np.array(result, dtype=np.uint8))
cv.imwrite("hsv.png", np.array(result2, dtype=np.uint8))
cv.imwrite("roi2.png", np.array(result, dtype=np.uint8))
cv.waitKey(0)    