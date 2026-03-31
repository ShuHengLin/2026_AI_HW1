import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

class StreetViewDataset(Dataset):
    def __init__(self, csv_file, split, transform=None, subset_ratio=1.0):
        df = pd.read_csv(csv_file)
        self.transform = transform
        self.data = df[df['split'] == split].reset_index(drop=True)
        if subset_ratio < 1.0:
            self.data = self.data.sample(frac=subset_ratio, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.data.iloc[idx]['label']
        return image, label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
CURRENT_RATIO = 1.0  # 0.2, 0.5, 1.0
USE_AUGMENTATION = True
if USE_AUGMENTATION:
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

print(f"正在執行實驗：使用 {int(CURRENT_RATIO*100)}% 的訓練數據")
train_set = StreetViewDataset("metadata.csv", split='train', transform=transform, subset_ratio=CURRENT_RATIO)
val_set   = StreetViewDataset("metadata.csv", split='val',   transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

# 定義模型，並修改最後一層全連接層
model = models.resnet18() # weights=models.ResNet18_Weights.DEFAULT
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 3)
)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

history = {
    'train_loss': [],
    'train_acc': [],
    'val_acc': []
}
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    # 驗證階段
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    history['val_acc'].append(val_acc)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {current_lr}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), history['train_loss'], label='Train Loss', color='red')
plt.title(f'Training Loss (Ratio: {CURRENT_RATIO})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), history['train_acc'], label='Train Acc', color='blue')
plt.plot(range(1, EPOCHS+1), history['val_acc'],   label='Val Acc',   color='green')
plt.title(f'Accuracy Curve (Ratio: {CURRENT_RATIO})')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig(f"learning_curve_ratio_{CURRENT_RATIO}.png")
print(f"曲線圖已儲存為 learning_curve_ratio_{CURRENT_RATIO}.png")

import json
aug_tag = "_aug" if USE_AUGMENTATION else ""
history_filename = f"history_ratio_{CURRENT_RATIO}{aug_tag}.json"
with open(history_filename, 'w') as f:
    json.dump(history, f)
print(f"訓練紀錄已儲存至 {history_filename}")

torch.save(model.state_dict(), f"resnet18_ratio_{CURRENT_RATIO}{aug_tag}.pth")
print(f"比例 {CURRENT_RATIO} 訓練完成。")