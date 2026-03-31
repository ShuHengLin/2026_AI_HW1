import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tqdm import tqdm
from PIL import Image
import pandas as pd
#

USE_GRAYSCALE = False
def load_data_as_vectors(csv_file, split_name, grayscale=True):
    df = pd.read_csv(csv_file)
    subset = df[df['split'] == split_name]
    features = []
    labels = []
    mode = 'L' if grayscale else 'RGB'
    print(f"正在載入 {split_name} 並轉換為向量...")
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        img = Image.open(row['file_path']).convert(mode).resize((64, 64))  # 讀取圖片並轉成 8-bit 灰階，再縮小至 64x64 以節省運算時間
        img_vector = np.array(img).flatten() / 255.0                       # 標準化到 0-1
        features.append(img_vector)
        labels.append(row['label'])
    return np.array(features), np.array(labels)

x_train, y_train = load_data_as_vectors("metadata.csv", "train", grayscale=USE_GRAYSCALE)
x_test,  y_test  = load_data_as_vectors("metadata.csv", "test",  grayscale=USE_GRAYSCALE)

# PCA 降維，將 64*64=4096 維降到 100 維
start_time = time.time()
DIM_REDUCTION = False
if DIM_REDUCTION:
    n_components = 100
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca  = pca.transform(x_test)
    print(f"PCA 降維完成：原始維度 {x_train.shape[1]} -> 降維後 {x_train_pca.shape[1]}")
else:
    x_train_pca = x_train
    x_test_pca  = x_test
    print(f"不使用降維 | 維度: {x_train.shape[1]}")

from sklearn.model_selection import cross_val_score

# SVM 模型與 Cross-validation 評估
print(f"正在進行 5-Fold Cross-validation (PCA: {DIM_REDUCTION})...")
cv_start = time.time()
svm = SVC(kernel='rbf', C=10, gamma='scale')
cv_scores = cross_val_score(svm, x_train_pca, y_train, cv=5)
cv_time = time.time() - cv_start
print(f"CV 完成！耗時: {cv_time:.2f}s")
print(f"各折準確度: {cv_scores}")
print(f"平均 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("正在訓練 SVM (RBF Kernel)...")
train_start = time.time()
svm.fit(x_train_pca, y_train)
train_end = time.time()
train_time = train_end - train_start

y_pred = svm.predict(x_test_pca)
test_time = time.time() - train_end
total_time = time.time() - start_time

# 計算指標
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
target_names = ['Taiwan', 'Japan', 'Iceland']
print(f"\n--- 實驗結果 (Gray: {USE_GRAYSCALE}, PCA: {DIM_REDUCTION}) ---")
print(f"5-Fold CV Mean Accuracy: {cv_scores.mean():.4f}") # 這是作業要求的指標
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Train time: {train_time:.2f}s")
print(f"Test time:  {test_time:.2f}s")
print(f"Total time: {total_time:.2f}s")
print("\n詳細分類報告：")
print(classification_report(y_test, y_pred, target_names=target_names))

# 繪製混淆矩陣圖
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - PCA + SVM Baseline')
plt.savefig("confusion_matrix_svm.png")