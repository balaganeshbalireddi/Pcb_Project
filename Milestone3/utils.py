# utils.py
import cv2
import numpy as np

def generate_mask(template_img, test_img):
    if template_img.shape != test_img.shape:
        template_img = cv2.resize(template_img, (test_img.shape[1], test_img.shape[0]))

    diff = cv2.absdiff(template_img, test_img)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff = cv2.GaussianBlur(diff, (7,7), 0)

    _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 150:
            cv2.drawContours(clean, [cnt], -1, 255, -1)

    return clean


def extract_rois(test_img, mask, pad=15):
    rois = []
    boxes = []

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, test_img.shape[1])
        y2 = min(y + h + pad, test_img.shape[0])

        roi = test_img[y1:y2, x1:x2]
        rois.append(roi)
        boxes.append((x1,y1,x2,y2))

    return rois, boxes
