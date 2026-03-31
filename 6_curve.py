import json
import matplotlib.pyplot as plt

experiments = [
    ("history_ratio_0.2.json", "20% Data", "tab:blue", "-"),
    ("history_ratio_0.5.json", "50% Data", "tab:orange", "-"),
    ("history_ratio_1.0.json", "100% Data", "tab:green", "-"),
    ("history_ratio_1.0_aug.json", "100% Data + Augmentation", "tab:red", "--") 
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for file_path, label_str, color, style in experiments:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        epochs = range(1, len(data['val_acc']) + 1)
        ax1.plot(epochs, data['val_acc'], label=label_str, color=color, 
                 linestyle=style, linewidth=2, marker='o', markersize=4)
        ax2.plot(epochs, data['train_loss'], label=label_str, color=color, 
                 linestyle=style, linewidth=2, marker='s', markersize=4)
    except FileNotFoundError:
        print(f"警告：找不到 {file_path}")

# 設定 Accuracy 圖表 (左)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax1.set_ylim(10, 100) 
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 設定 Loss 圖表 (右)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig("combined_experiment_analysis.png", dpi=300)
plt.show()
print("對照圖已儲存為 combined_experiment_analysis.png")