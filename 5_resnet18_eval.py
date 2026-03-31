import torch
import torch.nn as nn
#
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
#

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
BATCH_SIZE = 32
CURRENT_RATIO = 1.0  # 0.2, 0.5, 1.0
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_set = StreetViewDataset("metadata.csv", split='test', transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 定義模型，並修改最後一層全連接層
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 3)
)
model.load_state_dict(torch.load("resnet18_ratio_" + str(CURRENT_RATIO) + "_aug.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []
print("正在對測試集進行推論...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# 計算指標
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
target_names = ['Taiwan', 'Japan', 'Iceland']
print("\n--- 測試集評估結果 ---")
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print("\n詳細分類報告：")
print(classification_report(all_labels, all_preds, target_names=target_names))

# 繪製混淆矩陣圖
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Street View Classification (3 Classes)')
plt.savefig("confusion_matrix_" + str(CURRENT_RATIO) + ".png")