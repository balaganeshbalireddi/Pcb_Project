import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import cv2
import os

MODEL_PATH = "models/pcb_defect_model.pth"
INPUT_DIR = "data/raw_rois"
OUTPUT_DIR = "results/predicted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = ["missing_hole","mouse_bite","open_circuit","short","spur","spurious_copper"]

model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

for img_name in os.listdir(INPUT_DIR):
    img = cv2.imread(os.path.join(INPUT_DIR, img_name))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(rgb).unsqueeze(0)

    with torch.no_grad():
        pred = torch.argmax(model(x),1).item()
        label = classes[pred]

    cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)

print("🎯 Prediction images saved in results/predicted/")
