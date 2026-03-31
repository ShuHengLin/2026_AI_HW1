import torch
import torch.nn as nn
#
#
from torchvision import models, transforms
from PIL import Image
import pandas as pd
#

df = pd.read_csv("metadata.csv")
test_df = df[df['split'] == 'test']
random_row = test_df.sample(n=1).iloc[0]  # 隨機抽一行
image_path = random_row['file_path']
actual_label = random_row['country']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_RATIO = 1.0  # 0.2, 0.5, 1.0
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

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

img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device) # 增加 Batch 維度 (1, 3, 224, 224)

with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

classes = ['Taiwan', 'Japan', 'Iceland']
pred_label = classes[predicted.item()]
print(f"圖片路徑: {image_path}")
print(f"真實類別: {actual_label}")
print(f"預測結果: {pred_label}")
img.save("inference_result.jpg", "JPEG", quality=90)