import os
import pandas as pd
import numpy as np

categories = {
        "Taiwan":  0,
        "Japan":   1,
        "Iceland": 2
    }

def create_metadata_csv(root_dir, output_file="metadata.csv", seed=23):
    np.random.seed(seed)
    data_list = []

    for category_name, label in categories.items():
        category_path = os.path.join(root_dir, category_name)
        if not os.path.exists(category_path):
            print(f"警告：找不到資料夾 {category_path}，跳過...")
            continue

        for subdir, dirs, files in os.walk(category_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(subdir, file)
                    region_name = os.path.basename(subdir)
                    data_list.append({
                        "file_path": full_path,
                        "label": label,
                        "country": category_name,
                        "region": region_name
                    })

    # 轉成 DataFrame 並儲存
    df = pd.DataFrame(data_list)

    # 根據 region 分配 Train / Val / Test 標籤
    unique_regions = df[['country', 'region']].drop_duplicates().reset_index(drop=True)
    unique_regions = unique_regions.sample(frac=1, random_state=seed).reset_index(drop=True)
    num_regions = len(unique_regions)
    r_train_end = int(num_regions * 0.7)  # 70%
    r_val_end   = int(num_regions * 0.85) # 15% (70% + 15%)
    unique_regions['split'] = 'train'     # 預設全部為 train
    unique_regions.iloc[r_train_end:r_val_end, unique_regions.columns.get_loc('split')] = 'val'
    unique_regions.iloc[r_val_end:, unique_regions.columns.get_loc('split')] = 'test'

    df = df.merge(unique_regions[['country', 'region', 'split']], on=['country', 'region'], how='left')

    print(f"區域總數: {num_regions}")
    print(f"數據分配完成：\n{df['split'].value_counts().to_dict()}")
    print("-" * 30)
    print(df.groupby(['split', 'country']).size().to_string(dtype=False))
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"成功！已生成 {output_file}，共計 {len(df)} 張圖片。")

if __name__ == "__main__":
    create_metadata_csv("data")