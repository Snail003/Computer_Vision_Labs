import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def filter_by_hsv(img, ref_hsv=(0, 255, 255), h_tol=20, sv_tol=210):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    h, s, v = cv2.split(hsv_img)
    rh, rs, rv = ref_hsv
    diff_hue = np.minimum(np.abs(h - rh), 180 - np.abs(h - rh))
    dist = np.sqrt((diff_hue/h_tol)**2 + ((s-rs)/sv_tol)**2 + ((v-rv)/sv_tol)**2)
    m = (dist >= 1).astype(np.uint8) * 255
    cleaned = cv2.bitwise_and(img, img, mask=m)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(m, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cleaned)
    plt.title("Filtered")
    plt.axis("off")
    plt.show()
    return cleaned
def segment_kmeans(img, clusters=2):
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(data, clusters, None, criteria, 15, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Before segmentation")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title(f"K-Means = {clusters}")
    plt.axis("off")
    plt.show()
    return segmented
def detect_regions(img_for_processing, img_for_display, min_size=2, save_folder="Lab2/results"):
    gray = cv2.cvtColor(img_for_processing, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > min_size]
    result_count = len(valid)
    vis1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    vis2 = img_for_display.copy()
    cv2.drawContours(vis1, valid, -1, (0, 255, 0), 1)
    cv2.drawContours(vis2, valid, -1, (0, 255, 0), 1)
    plt.figure(figsize=(13, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(vis1)
    plt.title(f"Detected: {result_count}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(vis2)
    plt.title("Visualization")
    plt.axis("off")
    plt.show()
    os.makedirs(save_folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p1 = os.path.join(save_folder, f"proc_{ts}.png")
    p2 = os.path.join(save_folder, f"vis_{ts}.png")
    cv2.imwrite(p1, cv2.cvtColor(vis1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(p2, cv2.cvtColor(vis2, cv2.COLOR_RGB2BGR))
    return result_count, vis1, vis2

image_path = "Lab2/image.png"
img_raw = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
step1 = filter_by_hsv(img_rgb, sv_tol=210)
step2 = segment_kmeans(step1, clusters=2)
count_buildings, view1, view2 = detect_regions(step2, img_rgb)