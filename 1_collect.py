import requests
import os, time
from tqdm import tqdm
from PIL import Image, ImageOps
from io import BytesIO

ACCESS_TOKEN = "MLY|25647034868308444|ba8f11c140880252f09d7d02a0451bc9"
URBAN_POINTS = {
    "Taiwan": [
        # --- 台北/新北 (極高密度) ---
        [121.505, 25.043, "TW_Taipei_Ximen"],      # 西門商圈
        [121.544, 25.039, "TW_Taipei_Daan"],       # 大安商圈
        # --- 桃竹苗 ---
        [121.224, 24.954, "TW_Zhongli_Center"],    # 中壢站前
        [120.968, 24.805, "TW_Hsinchu_Temple"],    # 新竹城隍廟
        # --- 台中/彰化/南投 ---
        [120.685, 24.148, "TW_Taichung_Yizhong"],  # 台中一中街
        [120.663, 24.151, "TW_Taichung_West"],     # 台中勤美草悟道
        [120.542, 24.081, "TW_Changhua_Center"],   # 彰化市區
        [120.686, 23.906, "TW_Nantou_Center"],     # 南投市區
        # --- 雲嘉南 ---
        [120.449, 23.479, "TW_Chiayi_Wenhua"],     # 嘉義文化路
        [120.197, 22.993, "TW_Tainan_West"],       # 台南中西區
        # --- 高屏/東部 ---
        [120.302, 22.622, "TW_Kaohsiung_Central"], # 高雄新堀江
        [120.306, 22.637, "TW_Kaohsiung_Station"], # 高雄站前
        [121.606, 23.977, "TW_Hualien_Center"],    # 花蓮市區
        [121.150, 22.755, "TW_Taitung_Center"]     # 台東市區
    ],
    "Japan": [
        # --- 北日本 (北海道) ---
        [141.350, 43.059, "JP_Sapporo_Odori"],     # 札幌大通
        [141.353, 43.055, "JP_Sapporo_Susukino"],  # 札幌薄野
        [141.352, 43.051, "JP_Sapporo_Nakajima"],  # 札幌中島公園周邊
        # --- 中部 (北陸) ---
        [136.654, 36.561, "JP_Kanazawa_Korinbo"],  # 金澤香林坊
        # --- 關東 (東京・橫濱) ---
        [139.712, 35.731, "JP_Tokyo_Ikebukuro"],   # 池袋西口
        [139.699, 35.701, "JP_Tokyo_Okubo"],       # 大久保
        [139.771, 35.698, "JP_Tokyo_Akihabara"],   # 秋葉原
        [139.764, 35.672, "JP_Tokyo_Ginza"],       # 銀座
        [139.757, 35.666, "JP_Tokyo_Shinbashi"],   # 新橋
        [139.642, 35.444, "JP_Yokohama_Kannai"],   # 橫濱關內
        # --- 關西 (大阪・奈良) ---
        [135.530, 34.697, "JP_Osaka_Kyobashi"],    # 京橋
        [135.513, 34.686, "JP_Osaka_Tanimachi"],   # 大阪谷町
        [135.805, 34.683, "JP_Nara_SanjoDori"],    # 奈良三条通
        [135.502, 34.673, "JP_Osaka_Shinsaibashi"],# 心齋橋筋周邊
        [135.502, 34.669, "JP_Osaka_Dotonbori"],   # 道頓堀
        [135.501, 34.666, "JP_Osaka_Namba"],       # 難波
        [135.506, 34.652, "JP_Osaka_Shinsekai"],   # 新世界
        # --- 九州 ---
        [130.407, 33.585, "JP_Fukuoka_Tenjin_Minami"], # 福岡天神南
        [130.708, 32.803, "JP_Kumamoto_Torichosuji"],  # 熊本通町筋
    ],
    "Iceland": [
        # --- 雷克雅維克 (Reykjavík) ---
        [-21.928, 64.145, "IS_Rey_Laugavegur"],    # 主購物街
        [-21.932, 64.148, "IS_Rey_Rainbow_St"],    # 彩虹大道
        [-21.942, 64.147, "IS_Rey_Old_Town"],      # 舊城中心
        [-21.940, 64.150, "IS_Rey_Harpa_Area"],    # 現代建築區
        [-21.948, 64.152, "IS_Rey_Grandi"],        # 港口商圈
        [-21.954, 64.150, "IS_Rey_Old_Port"],      # 舊港口餐廳街
        [-21.924, 64.141, "IS_Rey_Church_Res"],    # 大教堂周邊住宅
        # --- 大雷克雅維克衛星城市 (Satellite Cities) ---
        [-21.952, 64.068, "IS_Hafnarfjordur"],      # 哈夫納夫約杜爾老街
        [-21.905, 64.070, "IS_Hafnarf_Commercial"], # 哈夫納夫約杜爾購物區?
        # --- 西部與西北部城鎮 (Western & Westfjords) ---
        [-22.557, 64.001, "IS_Keflavik_Street"],    # 凱夫拉維克主街
        # --- 北部最大城 (Akureyri) ---
        [-18.089, 65.681, "IS_Akureyri_Walking"],   # 阿庫雷里步行街
        [-18.092, 65.679, "IS_Akureyri_Center"],    # 阿庫雷里劇院周邊
        # --- 東部與南部沿海 (East & South Coast) ---
        [-14.015, 65.072, "IS_Neskaupstadur"],      # 東部峽灣密集小鎮
        [-15.212, 64.254, "IS_Hofn_Center"],        # 赫本港口商業區
    ]
}

def process_and_save_image(img_content, save_path):
    """將下載的二進位圖片裁切成正方形"""
    try:
        img = Image.open(BytesIO(img_content))
        img = ImageOps.exif_transpose(img) # 自動修正圖片方向
        img = img.convert('RGB')
        w, h = img.size
        min_dim = min(w, h)
        left, top = (w - min_dim) / 2, (h - min_dim) / 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img.save(save_path, "JPEG", quality=90)
        return True
    except:
        return False


def download_and_preprocess(country, target_total=5000):
    print(f"\n🚀 開始採集 {country} (目標: {target_total} 張)...")
    save_dir = f"data/{country}"
    os.makedirs(save_dir, exist_ok=True)

    points = URBAN_POINTS[country]
    imgs_per_point = 150

    # 初始化總數（掃描現有檔案）
    current_total = 0
    for root, dirs, files in os.walk(save_dir):
        current_total += len([f for f in files if f.endswith('.jpg')])

    for lon, lat, place_name in points:
        if current_total >= target_total:
            break

        # 為每個地點建立專屬子資料夾：data/Taiwan/Ximending/
        point_dir = f"data/{country}/{place_name}"
        os.makedirs(point_dir, exist_ok=True)

        d = 0.003 # 搜尋半徑 (0.003 約方圓 300 公尺)，確保鎖定在商圈核心
        bbox = f"{lon-d},{lat-d},{lon+d},{lat+d}"
        url = f"https://graph.mapillary.com/images?access_token={ACCESS_TOKEN}&bbox={bbox}&fields=id,thumb_1024_url&limit={imgs_per_point}"
        try:
            res = requests.get(url, timeout=15).json()
            data = res.get('data', [])
            if not data:
                continue
            print(f" 📍 座標 ({lat}, {lon}) 找到 {len(data)} 張圖...")
            pbar = tqdm(data, desc=f"下載 {place_name}", leave=False)
            for img in pbar:
                file_path = f"{point_dir}/{img['id']}.jpg"
                if os.path.exists(file_path):
                    continue
                try:
                    img_res = requests.get(img['thumb_1024_url'], timeout=10)
                    if process_and_save_image(img_res.content, file_path):
                        current_total += 1
                except:
                    continue
                # 如果該點已經抓夠了，或是總數達標就跳出
                if current_total >= target_total:
                    break
            time.sleep(0.2)
        except Exception as e:
            print(f" ⚠️ {place_name} 發生錯誤: {e}")
            continue
    print(f"✨ {country} 採集完成！資料夾位於 data/{country}/ 下各子目錄。")


if __name__ == "__main__":
    download_and_preprocess("Taiwan")
    download_and_preprocess("Japan")
    download_and_preprocess("Iceland")
    print("\n" + "="*30)
    for c in ["Taiwan", "Japan", "Iceland"]:
        total = sum(len([f for f in files if f.endswith('.jpg')]) for _, _, files in os.walk(f"data/{c}"))
        print(f"{c} 資料夾總計圖片: {total} 張")