import cv2
import os
import numpy as np

MASK_DIR = "outputs/masks"
TEST_DIR = "data/test"
OUT_ANN = "outputs/annotated"
OUT_ROI = "outputs/rois"

os.makedirs(OUT_ANN, exist_ok=True)
os.makedirs(OUT_ROI, exist_ok=True)

for img_name in os.listdir(MASK_DIR):

    if not img_name.endswith(".jpg"):
        continue

    mask = cv2.imread(os.path.join(MASK_DIR, img_name), cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(os.path.join(TEST_DIR, img_name))

    if mask is None or test_img is None:
        continue

    # Connect thin defects
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    roi_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        pad = 15
        x = max(x - pad, 0)
        y = max(y - pad, 0)
        w = min(w + 2 * pad, test_img.shape[1] - x)
        h = min(h + 2 * pad, test_img.shape[0] - y)

        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        roi = test_img[y:y+h, x:x+w]
        cv2.imwrite(
            os.path.join(OUT_ROI, f"{img_name[:-4]}_roi_{roi_count}.jpg"),
            roi
        )

        roi_count += 1

    cv2.imwrite(os.path.join(OUT_ANN, img_name), test_img)
    print(f"✅ {img_name} → {roi_count} ROIs")

print("\n🎯 Module 2 COMPLETED (ROI EXTRACTION DONE)")