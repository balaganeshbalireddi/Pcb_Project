import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- TRANSFORMS ----------------
train_tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

test_tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---------------- DATA ----------------
train_data = ImageFolder(TRAIN_DIR, transform=train_tf)
test_data = ImageFolder(TEST_DIR, transform=test_tf)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)

# ---------------- MODEL ----------------
model = torchvision.models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------- FOCAL LOSS ----------------
def focal_loss(inputs, targets, alpha=1, gamma=2):
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ---------------- TRAINING ----------------
epochs = 35
train_accs, test_accs = [], []
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    correct, total, train_loss = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = focal_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(outputs,1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_losses.append(train_loss/len(train_loader))
    train_accs.append(train_acc)

    # -------- TEST --------
    model.eval()
    correct, total, test_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = focal_loss(outputs, labels)

            test_loss += loss.item()
            _, pred = torch.max(outputs,1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total
    test_losses.append(test_loss/len(test_loader))
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "pcb_defect_model.pth"))

# ---------------- PLOTS ----------------
plt.figure()
plt.plot(train_accs,label="Train Accuracy")
plt.plot(test_accs,label="Test Accuracy")
plt.legend()
plt.savefig("results/accuracy.png")

plt.figure()
plt.plot(train_losses,label="Train Loss")
plt.plot(test_losses,label="Test Loss")
plt.legend()
plt.savefig("results/loss.png")

print("\n🎯 Training complete — model + graphs saved")
