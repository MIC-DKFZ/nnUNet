#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PNG Mask 转换脚本
功能：将 AISD 的 PNG 标注切片堆叠为 3D NIfTI
要求：必须与 CT 使用完全相同的 Spacing（从 spacing_info.json 读取）
输出：nnUNet_raw/labelsTr/xxx.nii.gz
"""

import os
import re
import json
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm


def natural_sort_key(s):
    """自然排序（处理 000.png, 001.png...）"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def convert_single_mask(case_id, mask_base_dir, spacing_info, output_dir):
    """
    转换单个病例的 PNG Mask
    """
    # 检查是否有对应的 spacing 信息
    if case_id not in spacing_info:
        print(f"  ✗ {case_id}: 未找到对应的 CT spacing 信息，请先转换 CT")
        return False

    info = spacing_info[case_id]
    mask_folder = Path(mask_base_dir) / case_id

    if not mask_folder.exists():
        print(f"  ⚠ {case_id}: 未找到 mask 文件夹 {mask_folder}")
        return False

    try:
        # 获取所有 PNG 并排序
        png_files = [f for f in os.listdir(mask_folder) if f.lower().endswith('.png')]
        png_files.sort(key=natural_sort_key)

        if len(png_files) == 0:
            print(f"  ✗ {case_id}: mask 文件夹中无 PNG 文件")
            return False

        print(f"  找到 {len(png_files)} 张 PNG 标注")

        # 读取第一张获取尺寸
        first_img = Image.open(mask_folder / png_files[0])
        width, height = first_img.size

        # 创建 3D 数组 (深度, 高度, 宽度)
        depth = len(png_files)
        mask_volume = np.zeros((depth, height, width), dtype=np.uint8)

        # 逐张读取并二值化
        for i, png_file in enumerate(png_files):
            img_path = mask_folder / png_file
            img = Image.open(img_path).convert('L')  # 转灰度
            arr = np.array(img)

            # 关键：PNG 值通常是 0 和 255，转为 0 和 1
            # 如果已经是 0/1，这步也无害
            if arr.max() > 1:
                mask_volume[i] = (arr > 0).astype(np.uint8)
            else:
                mask_volume[i] = arr

        # 创建 SimpleITK 图像
        mask_itk = sitk.GetImageFromArray(mask_volume)

        # 关键：应用与 CT 完全相同的 Spacing/Origin/Direction
        mask_itk.SetSpacing(tuple(info["spacing"]))
        mask_itk.SetOrigin(tuple(info["origin"]))
        mask_itk.SetDirection(tuple(info["direction"]))

        # 保存
        output_path = Path(output_dir) / "labelsTr" / f"{case_id}.nii.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(mask_itk, str(output_path))

        unique_labels = np.unique(mask_volume)
        print(f"  ✓ {case_id}: 标签值 {unique_labels}, 已保存至 {output_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ {case_id}: 转换失败 - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # ===== 用户配置区域 =====
    # 必须与 convert_dicom_to_nifti.py 中使用的一致
    mask_base_dir = Path(r"C:\Users\lenovo\Desktop\ASID数据集\mask")  # AISD/mask 文件夹位置（
    output_dir = Path(r"C:\Users\lenovo\Desktop\nnUNet\nnUNet_raw\Dataset001_AISDmini")  # 输出目录

    # 读取之前生成的 spacing 信息
    spacing_info_path = output_dir / "spacing_info.json"

    # 指定要处理的病例 ID（必须与 CT 转换的列表一致）
    case_ids = ["0091440", "0091519"]  # ← 必须与脚本 1 一致
    # 或自动检测所有：case_ids = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]

    # =======================

    # 检查 spacing_info.json 是否存在
    if not spacing_info_path.exists():
        print(f"✗ 错误: 未找到 {spacing_info_path}")
        print("请先运行 convert_dicom_to_nifti.py 转换 CT 并生成 spacing 信息")
        return

    # 读取 spacing 信息
    with open(spacing_info_path, 'r', encoding='utf-8') as f:
        spacing_info = json.load(f)

    print(f"将处理 {len(case_ids)} 个病例的 Mask 标注")
    print(f"Mask 目录: {mask_base_dir}")
    print(f"输出目录: {output_dir / 'labelsTr'}")
    print("-" * 60)

    # 转换所有 mask
    success_count = 0
    for case_id in tqdm(case_ids, desc="转换进度"):
        if convert_single_mask(case_id, mask_base_dir, spacing_info, output_dir):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"✓ 完成！成功转换 {success_count} 个 mask")
    print(f"✓ 标注保存在: {output_dir / 'labelsTr'}")

    # 检查图像和标注是否配对
    images = list((output_dir / "imagesTr").glob("*.nii.gz"))
    labels = list((output_dir / "labelsTr").glob("*.nii.gz"))
    print(f"  图像文件数: {len(images)}")
    print(f"  标注文件数: {len(labels)}")

    if len(images) != len(labels):
        print(f"  ⚠ 警告: 图像和标注数量不匹配！")


if __name__ == "__main__":
    main()