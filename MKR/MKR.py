import cv2
import numpy as np

img = cv2.imread("earth.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

sat_boost = 3
s = np.clip(s.astype(np.float32) * sat_boost, 0, 255).astype(np.uint8)

hsv_boosted = cv2.merge([h, s, v])
img_saturated = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

img_blur = cv2.GaussianBlur(img_saturated, (7, 7), 2)

H0, S0, V0 = 120.0, 200.0, 200.0

ref_hsv = np.uint8([[[int(H0), int(S0), int(V0)]]])
ref_bgr = cv2.cvtColor(ref_hsv, cv2.COLOR_HSV2BGR)[0, 0]

ref_patch = np.full((100, 100, 3), ref_bgr, dtype=np.uint8)
cv2.imwrite("reference_color_patch.png", ref_patch)

hsv2 = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV).astype(np.float32)

H = hsv2[:, :, 0]
S = hsv2[:, :, 1]
V = hsv2[:, :, 2]

dH = H - H0
dS = S - S0
dV = V - V0

dist_sq = dH**2 + dS**2 + dV**2

threshold = 90.0
threshold_sq = threshold**2

blue_mask = (dist_sq < threshold_sq).astype(np.uint8) * 255

gray = cv2.cvtColor(img_saturated, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

mask_3ch = cv2.merge([blue_mask, blue_mask, blue_mask]).astype(np.float32) / 255.0

color_part = img_saturated.astype(np.float32) * mask_3ch
gray_part  = gray_bgr.astype(np.float32) * (1.0 - mask_3ch)

result = np.clip(color_part + gray_part, 0, 255).astype(np.uint8)

cv2.imwrite("step1.jpg", img_saturated)
cv2.imwrite("step2.jpg", img_blur)
cv2.imwrite("step3.jpg", blue_mask)
cv2.imwrite("step4.jpg", result)