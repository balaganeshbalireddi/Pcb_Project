import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PATHS ----------------
DATA_DIR = "data"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "models/pcb_defect_model.pth"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---------------- LOAD TEST DATA ----------------
test_data = ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
class_names = test_data.classes

# ---------------- LOAD MODEL ----------------
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---------------- RUN INFERENCE ----------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

# ---------------- METRICS ----------------
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)
accuracy = accuracy_score(y_true, y_pred)

print("\nClassification Report:\n", report)
print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")

# Save report
with open(os.path.join(RESULT_DIR, "report.txt"), "w") as f:
    f.write(f"Overall Test Accuracy: {accuracy*100:.2f}%\n\n")
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# ---------------- NEW: TRUE vs PRED CSV (MODULE 4) ----------------
rows = []

# test_data.samples gives (image_path, true_label)
for (img_path, true_label), pred_label in zip(test_data.samples, y_pred):
    rows.append({
        "image": os.path.basename(img_path),
        "true_label": class_names[true_label],
        "predicted_label": class_names[pred_label]
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(RESULT_DIR, "eval_comparison.csv")
df.to_csv(csv_path, index=False)

print(f"📄 Saved comparison table: {csv_path}")
print("🎯 Confusion matrix, report, and CSV saved in results/")
