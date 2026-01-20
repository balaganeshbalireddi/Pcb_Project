import cv2
import os
import numpy as np

TEMPLATE_DIR = "data/template"
TEST_DIR = "data/test"
OUT_MASK = "outputs/masks"

os.makedirs(OUT_MASK, exist_ok=True)

templates = os.listdir(TEMPLATE_DIR)

for img_name in os.listdir(TEST_DIR):

    if not img_name.endswith(".jpg"):
        continue

    # 🔹 Extract board ID (01, 04, etc.)
    board_id = img_name.split("_")[0]
    template_path = os.path.join(TEMPLATE_DIR, f"{board_id}.jpg")

    if not os.path.exists(template_path):
        print(f"⚠️ Template missing for {img_name}")
        continue

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(os.path.join(TEST_DIR, img_name), cv2.IMREAD_GRAYSCALE)

    if template is None or test_img is None:
        continue

    if template.shape != test_img.shape:
        template = cv2.resize(template, (test_img.shape[1], test_img.shape[0]))

    # 1️⃣ Absolute difference
    diff = cv2.absdiff(template, test_img)

    # 2️⃣ Normalize contrast
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 3️⃣ Blur
    diff = cv2.GaussianBlur(diff, (7, 7), 0)

    # 4️⃣ Threshold
    _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # 5️⃣ Morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6️⃣ Area filtering
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) > 150:
            cv2.drawContours(clean, [cnt], -1, 255, -1)

    cv2.imwrite(os.path.join(OUT_MASK, img_name), clean)
    print(f"✅ Correct mask generated: {img_name}")

print("\n🎯 Module 1 COMPLETED (DATASET-CORRECT)")