import os
import shutil
import json
import random
import re
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from psd_tools import PSDImage
from tqdm import tqdm

# 設定
source_image_path = '/Users/yuma/Yuma-Kanematsu/nnUNet/src/raw_data/Photoshop_annotation_data/Trap_door_muscle_fracture'
source_label_path = '/Users/yuma/Yuma-Kanematsu/nnUNet/src/raw_data/250328_for_test'
output_base_path = '/Users/yuma/Yuma-Kanematsu/nnUNet/nnUNet_raw'
dataset_id = 1
dataset_name = 'ForTest'
test_patient_num = 2  # テスト用に選ぶ患者数
color_threshold = 50  # 赤色チャンネルに対する閾値

# PSDを赤色ベースのマスクに変換する関数
def convert_psd_to_red_mask(psd_path, color_threshold=50):
    """
    PSDファイルを読み込み、赤色チャンネルをベースにした二値マスクを生成する
    """
    try:
        # PSDファイルを開く
        psd = PSDImage.open(psd_path)
        # PSDイメージを合成画像として取得
        composite_image = psd.composite()
        # numpy配列に変換
        img_array = np.array(composite_image)
        
        # RGBAの配列から赤色チャンネルを抽出
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBAの場合
            red_channel = img_array[:, :, 0]
            green_channel = img_array[:, :, 1]
            blue_channel = img_array[:, :, 2]
            alpha_channel = img_array[:, :, 3]
            
            # 赤色マスクの条件: 赤成分が高く、緑と青が低い、透明度がある
            red_mask = ((red_channel > green_channel + color_threshold) & 
                        (red_channel > blue_channel + color_threshold) & 
                        (alpha_channel > 0))
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGBの場合
            red_channel = img_array[:, :, 0]
            green_channel = img_array[:, :, 1]
            blue_channel = img_array[:, :, 2]
            
            # 赤色マスクの条件
            red_mask = ((red_channel > green_channel + color_threshold) & 
                        (red_channel > blue_channel + color_threshold))
        else:
            # グレースケールまたは対応していない形式
            print(f"警告: 対応していない画像形式です（shape: {img_array.shape}）")
            return None
        
        # 二値マスクに変換（True = 1, False = 0）- nnU-Netでは通常1が前景
        binary_mask = red_mask.astype(np.uint8)
        
        return binary_mask
    except Exception as e:
        print(f"エラー: PSDファイル {psd_path} の処理中にエラーが発生しました: {str(e)}")
        return None

# フルパス
dataset_folder = f'Dataset{dataset_id:03d}_{dataset_name}'
output_path = os.path.join(output_base_path, dataset_folder)

# 出力ディレクトリ作成
os.makedirs(os.path.join(output_path, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'imagesTs'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'labelsTr'), exist_ok=True)

# 患者IDの抽出
def extract_patient_id(filename):
    # 6375522_20_R_FLOOR-3.jpg のようなファイル名から患者IDを抽出
    match = re.match(r'(\d+)_\d+', filename)
    if match:
        return match.group(1)
    return None

# すべての画像ファイルを取得
all_image_files = [f for f in os.listdir(source_image_path) if f.endswith('.jpg')]
all_label_files = [f for f in os.listdir(source_label_path) if f.endswith('.psd')]

# 患者IDごとにファイルをグループ化
patient_files = {}
for img_file in all_image_files:
    patient_id = extract_patient_id(img_file)
    if patient_id:
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(img_file)

# ユニークな患者IDのリスト
patient_ids = list(patient_files.keys())
print(f"検出された患者数: {len(patient_ids)}")
print(f"患者ID: {patient_ids}")

# テスト用に患者をランダムに選択
if test_patient_num > len(patient_ids):
    test_patient_num = len(patient_ids) // 2  # 全体の半分をテスト用に
    print(f"指定された患者数が全体数を超えています。テスト用患者数を{test_patient_num}に調整します。")

test_patient_ids = random.sample(patient_ids, test_patient_num)
print(f"テスト用に選択された患者ID: {test_patient_ids}")

# ファイルの処理と変換
training_cases = []
label_classes = set()

# 進捗表示
total_cases = sum(len(patient_files[pid]) for pid in patient_ids)
print(f"合計 {total_cases} ケースを処理します...")

# 処理カウンター
processed_count = 0
success_count = 0
skip_count = 0
error_count = 0

for patient_id in patient_ids:
    is_test = patient_id in test_patient_ids
    
    for img_file in patient_files[patient_id]:
        processed_count += 1
        print(f"処理中: {processed_count}/{total_cases} - {img_file}")
        
        # ファイル名から必要な情報を抽出
        base_name = os.path.splitext(img_file)[0]
        
        # 対応するラベルファイルが存在するか確認
        label_file = f"{base_name}.psd"
        if label_file not in all_label_files:
            print(f"警告: {img_file}に対応するラベルファイル{label_file}が見つかりません。スキップします。")
            skip_count += 1
            continue
            
        # 画像ファイルを読み込み、PNGに変換
        img_path = os.path.join(source_image_path, img_file)
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"エラー: 画像ファイル {img_path} の読み込み中にエラーが発生しました: {str(e)}")
            error_count += 1
            continue
        
        # 保存先ディレクトリ
        if is_test:
            target_dir = os.path.join(output_path, 'imagesTs')
        else:
            target_dir = os.path.join(output_path, 'imagesTr')
            training_cases.append(base_name)
        
        # nnU-Net形式に合わせてファイル名を変更して保存
        target_img_path = os.path.join(target_dir, f"{base_name}_0000.png")
        img.save(target_img_path, format='PNG')
        
        # トレーニングデータの場合はラベルも処理
        if not is_test:
            # PSDファイルを赤色マスクに変換
            label_path = os.path.join(source_label_path, label_file)
            label_array = convert_psd_to_red_mask(label_path, color_threshold)
            
            if label_array is not None:
                # マスクに何か検出されたか確認
                if np.any(label_array > 0):
                    label_classes.add(1)  # クラス1としてカウント
                    
                    # ラベルを保存
                    label_img = Image.fromarray(label_array * 255)  # 表示用に255に拡大
                    target_label_path = os.path.join(output_path, 'labelsTr', f"{base_name}.png")
                    label_img.save(target_label_path, format='PNG')
                    success_count += 1
                else:
                    print(f"警告: {label_file} からマスクが検出されませんでした。")
                    # 空のマスクを保存
                    empty_mask = np.zeros(img.size[::-1], dtype=np.uint8)
                    label_img = Image.fromarray(empty_mask)
                    target_label_path = os.path.join(output_path, 'labelsTr', f"{base_name}.png")
                    label_img.save(target_label_path, format='PNG')
                    skip_count += 1
            else:
                print(f"エラー: {label_file} からマスクを生成できませんでした。")
                error_count += 1

# dataset.json ファイルの作成
dataset_json = {
    "channel_names": {
        "0": "Image",  # モダリティ名を設定
    },
    "labels": {
        "background": 0,
        "fracture": 1,  # 赤色マスクをfractureとして扱う
    },
    "numTraining": len(training_cases),
    "file_ending": ".png"
}

# dataset.json を保存
with open(os.path.join(output_path, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print("\n処理結果:")
print(f"データセットの作成が完了しました。場所: {output_path}")
print(f"トレーニングケース数: {len(training_cases)}")
print(f"成功: {success_count}, スキップ: {skip_count}, エラー: {error_count}")
print(f"ラベルクラス: {dataset_json['labels']}")
print(f"注意: PSDファイルから赤色ベースのマスクを抽出しています。")
print(f"      色閾値を変更する場合は color_threshold パラメータを調整してください（現在: {color_threshold}）。")