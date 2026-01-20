# inference.py  — MILESTONE 4 / MODULE 7 BACKEND

import torch
torch.set_num_threads(4)     # <-- SPEED OPTIMIZATION

import torchvision
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn
from utils import generate_mask, extract_rois

MODEL_PATH = "models/pcb_defect_model.pth"

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

# Load model once (fast inference)
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def process_and_predict(template_img, test_img):

    logs = {}

    gray_t = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    mask = generate_mask(gray_t, gray_test)
    logs["mask_generated"] = True

    rois, boxes = extract_rois(test_img, mask)
    logs["num_rois"] = len(rois)

    annotated = test_img.copy()
    predictions = []

    for roi, (x1,y1,x2,y2) in zip(rois, boxes):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        x = transform(rgb).unsqueeze(0)

        with torch.no_grad():
            pred = torch.argmax(model(x),1).item()

        label = CLASSES[pred]
        predictions.append(label)

        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(annotated, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    logs["predictions"] = predictions

    return mask, annotated, predictions, logs
