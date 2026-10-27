import cv2 as cv
import numpy as np
img =cv.imread('/home/dhruv/Desktop/Photos/road1.jpg')

#/home/dhruv/Desktop/Photos/road8.jpg
#/home/dhruv/Downloads/lane.jpeg




blur=cv.GaussianBlur(img, (5,5), 0)
canny = cv.Canny(blur,125,175)

resized= cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)   
white = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 40, 255]))
yellow = cv.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))



def region_of_interest(image, polygon):
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(image, mask)

def fit_curve(segs, degree=2):
    pts = np.array([p for s in segs for p in s[1]], dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]
    deg = min(degree, len(np.unique(ys)) - 1)    
    return np.polyfit(ys, xs, deg), ys.min()



def curved_polygon(left, right, h, w, margin=0.03, n=15):
    cl, top_l = fit_curve(left)
    cr, top_r = fit_curve(right)
    y_top = int(max(top_l, top_r))            
    pad = margin * w

    ys = np.linspace(y_top, h, n)           
    xl = np.clip(np.polyval(cl, ys) - pad, 0, w - 1)
    xr = np.clip(np.polyval(cr, ys) + pad, 0, w - 1)

    left_edge  = np.stack([xl, ys], axis=1)               
    right_edge = np.stack([xr, ys], axis=1)[::-1]        
    poly = np.vstack([left_edge, right_edge])
    return poly.astype(np.int32).reshape(1, -1, 2)

def road_region(img, tol_s=40, tol_v=45, close_size=31):
    h, w = img.shape[:2]
    hsv = cv.cvtColor(cv.GaussianBlur(img, (7, 7), 0), cv.COLOR_BGR2HSV)

    
    patch = hsv[int(.88 * h):int(.97 * h), int(.40 * w):int(.60 * w)].reshape(-1, 3)
    s_med = int(np.median(patch[:, 1]))
    v_med = int(np.median(patch[:, 2]))

    
    lo = np.array([0, 0, max(v_med - tol_v, 0)])
    hi = np.array([179, min(s_med + tol_s, 255), min(v_med + tol_v, 255)])
    asphalt = cv.inRange(hsv, lo, hi)

    
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_size, close_size))
    asphalt = cv.morphologyEx(asphalt, cv.MORPH_CLOSE, k)
    asphalt = cv.morphologyEx(asphalt, cv.MORPH_OPEN,
                              cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))

    
    n, labels = cv.connectedComponents(asphalt)
    seed_label = labels[int(.93 * h), w // 2]
    if seed_label == 0:
        return None
    road = np.uint8(labels == seed_label) * 255

    # 5. fill any remaining holes by redrawing the outer contour filled
    cnts, _ = cv.findContours(road, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(road)
    cv.drawContours(filled, cnts, -1, 255, cv.FILLED)
    return filled


# def auto_polygon(img, margin=0.05):
#     h, w = img.shape[:2]
#     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#     white = cv.inRange(hsv, np.array([0, 0, 200]), np.array([179, 40, 255]))
#     yellow = cv.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))
#     mask = cv.bitwise_or(white, yellow)

#     rough = np.array([[(0, h), (int(.3 * w), int(.55 * h)),
#                        (int(.7 * w), int(.55 * h)), (w, h)]], dtype=np.int32)
#     mask = region_of_interest(mask, rough)

#     edges = cv.Canny(cv.GaussianBlur(mask, (5, 5), 0), 50, 150)
#     lines = cv.HoughLinesP(edges, 1, np.pi / 180, 40,
#                            minLineLength=int(0.05 * h),
#                            maxLineGap=int(0.05 * w))
#     if lines is None:
#         return None
#     lines = lines.reshape(-1, 4)

#     left, right = [], []
#     for x1, y1, x2, y2 in lines:
#         if x1 == x2:
#             continue
#         m = (y2 - y1) / (x2 - x1)
#         if abs(m) < 0.3 or abs(m) > 3:
#             continue
#         xmid = (x1 + x2) / 2
#         seg = (m, [(x1, y1), (x2, y2)])
#         if m < 0 and xmid < w * 0.55:
#             left.append(seg)
#         elif m > 0 and xmid > w * 0.45:
#             right.append(seg)
#     if not left or not right:
#         return None

#     return curved_polygon(left, right, h, w, margin)      

# polygon = auto_polygon(img)
# if polygon is None:
#     print("auto polygon failed, using fallback")
#     h, w = img.shape[:2]
#     polygon = np.array([[(0, h), (int(.35*w), int(.5*h)),
#                          (int(.65*w), int(.5*h)), (w, h)]], dtype=np.int32)
# else:
#     print("auto polygon worked")
road = road_region(img)
if road is None:
    print("road region failed, using fallback")
    h, w = img.shape[:2]
    fallback = np.array([[(0, h), (int(.35*w), int(.5*h)),
                          (int(.65*w), int(.5*h)), (w, h)]], dtype=np.int32)
    road = np.zeros((h, w), np.uint8)
    cv.fillPoly(road, fallback, 255)
else:
    print("road region found")


cnts, _ = cv.findContours(road, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
preview = img.copy()
cv.drawContours(preview, cnts, -1, (0, 0, 255), 3)


near_road = cv.dilate(road, np.ones((25, 25), np.uint8))
lane = cv.bitwise_or(white, yellow)
roi = cv.bitwise_and(lane, near_road)
roi_result = cv.bitwise_and(img, img, mask=roi)
hsv_result = cv.bitwise_and(img, img, mask=lane)

cv.imshow("road mask", road)
# preview = img.copy()
# cv.polylines(preview, polygon, True, (0, 0, 255), 3)
   

# lane = cv.bitwise_or(white, yellow)
# roi = region_of_interest(lane, polygon)
# roi_result = cv.bitwise_and(img, img, mask=roi)
# hsv_result = cv.bitwise_and(img, img, mask=lane)

cv.imshow("Canny edges", canny )
cv.imshow("blur", blur)
cv.imshow("img",img)
cv.imshow("resized", resized)
cv.imshow("polygon preview", preview)
cv.imshow("hsv", hsv_result)
cv.imshow("roi", roi_result)
cv.imshow("road mask", road)


cv.imwrite("IMAGE.jpg", img)
cv.imwrite("HSV.jpg", np.array(hsv_result, dtype=np.uint8))
cv.imwrite("GAUSSIAN_BLUR.jpg", blur )
cv.imwrite("CANNY_EDGE.jpg", canny )

cv.imwrite("ROI.jpg", np.array(roi_result, dtype=np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()

