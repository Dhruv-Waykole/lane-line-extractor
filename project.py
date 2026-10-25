import cv2 as cv
import numpy as np
img =cv.imread('/home/dhruv/Desktop/Photos/road1.jpg')


def rescaleimage(frame, scale=1):
    width=int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


blur=cv.GaussianBlur(img, (5,5), 0)
canny = cv.Canny(blur,125,175)

resized= cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)   
white_mask = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 40, 255]))
yellow_mask = cv.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))




def region_of_interest(image, polygon):
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(image, mask)
def fit_side(segs):
   
    med = np.median([s[0] for s in segs])
    segs = [s for s in segs if abs(s[0] - med) < 0.3]
    if not segs:
        return None
    pts = np.array([p for s in segs for p in s[1]], dtype=np.float32)
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_HUBER, 0, 0.01, 0.01).flatten()
    m = vy / vx
    return m, y0 - m * x0

def auto_polygon(img, margin=0.03, top_frac=0.05):
    h, w = img.shape[:2]

    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    white = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 40, 255]))
    yellow = cv.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))
    mask = cv.bitwise_or(white, yellow)

    
    rough = np.array([[(0, h), (int(.3 * w), int(.55 * h)),
                       (int(.7 * w), int(.55 * h)), (w, h)]], dtype=np.int32)
    mask = region_of_interest(mask, rough)

    
    edges = cv.Canny(cv.GaussianBlur(mask, (5, 5), 0), 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 40,
                           minLineLength=int(0.05 * h),
                           maxLineGap=int(0.05 * w))

    
    if lines is None:
        return None
    lines = lines.reshape(-1, 4)

    
    left, right = [], []
    for x1, y1, x2, y2 in lines:
        if x1 == x2:
            continue
        m = (y2 - y1) / (x2 - x1)
        if abs(m) < 0.3 or abs(m) > 3:
            continue
        xmid = (x1 + x2) / 2
        seg = (m, [(x1, y1), (x2, y2)])
        if m < 0 and xmid < w * 0.55:
            left.append(seg)
        elif m > 0 and xmid > w * 0.45:
            right.append(seg)
    if not left or not right:
        return None

   
    fl, fr = fit_side(left), fit_side(right)
    if fl is None or fr is None:
        return None
    (ml, bl), (mr, br) = fl, fr

    
    xv = (br - bl) / (ml - mr)
    yv = ml * xv + bl
    if not (0.25 * h < yv < 0.75 * h) or not (0 < xv < w):
        return None
    y_top = int(yv + top_frac * (h - yv))

    x_at = lambda m, b, y: (y - b) / m
    pad = margin * w
    return np.array([[
        (int(x_at(ml, bl, h) - pad), h),
        (int(x_at(ml, bl, y_top) - pad), y_top),
        (int(x_at(mr, br, y_top) + pad), y_top),
        (int(x_at(mr, br, h) + pad), h),
    ]], dtype=np.int32)

polygon = auto_polygon(img)
preview = img.copy()
cv.polylines(preview, polygon, True, (0, 0, 255), 3)
if polygon is None:                      
    h, w = img.shape[:2]
    polygon = np.array([[(0, h), (int(.35*w), int(.5*h)),
                         (int(.65*w), int(.5*h)), (w, h)]], dtype=np.int32)

lane_mask = cv.bitwise_or(white_mask, yellow_mask)
roi_mask = region_of_interest(lane_mask, polygon)
roi_result = cv.bitwise_and(img, img, mask=roi_mask)
hsv_result = cv.bitwise_and(img, img, mask=lane_mask)

cv.imshow("Canny edges", canny )
cv.imshow("blur", blur)
cv.imshow("img",img)
cv.imshow("resized", resized)
cv.imshow("hsv", hsv_result)
cv.imshow("roi", roi_result)
cv.imshow("polygon preview", preview)

cv.imwrite("IMAGE.jpg", img)
cv.imwrite("HSV.jpg", np.array(hsv_result, dtype=np.uint8))
cv.imwrite("GAUSSIAN_BLUR.jpg", blur )
cv.imwrite("CANNY_EDGE.jpg", canny )
cv.imwrite("POLYGON.jpg", preview )
cv.imwrite("ROI.jpg", np.array(roi_result, dtype=np.uint8))

cv.waitKey(0)

