#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DICOM CT 转换脚本
功能：将 AISD 的 DICOM 序列转为 nnUNet 格式的 NIfTI
输出：nnUNet_raw/imagesTr/xxx_0000.nii.gz
      同时生成 spacing_info.json 供 mask 转换使用
"""

import os
import json
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm


def convert_single_case(case_id, base_dir, output_dir):
    """
    转换单个病例的 DICOM CT
    """
    dicom_folder = Path(base_dir) / case_id / "CT"
    output_path = Path(output_dir) / "imagesTr" / f"{case_id}_0000.nii.gz"

    if not dicom_folder.exists():
        print(f"  ✗ {case_id}: 未找到 CT 文件夹")
        return None

    try:
        # 读取 DICOM 序列
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))

        if len(dicom_names) == 0:
            print(f"  ✗ {case_id}: 文件夹中无 DICOM 文件")
            return None

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # 获取关键信息
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        size = image.GetSize()

        # 保存为 NIfTI
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_path))

        info = {
            "case_id": case_id,
            "spacing": list(spacing),  # [x, y, z]
            "origin": list(origin),
            "direction": list(direction),
            "size": list(size),
            "output_path": str(output_path)
        }

        print(f"  ✓ {case_id}: Spacing={spacing}, Size={size}")
        return info

    except Exception as e:
        print(f"  ✗ {case_id}: 转换失败 - {e}")
        return None


def main():
    # ===== 用户配置区域 =====
    base_dir = Path(r"C:\Users\lenovo\Desktop\ASID数据集")  # AISD 数据根目录
    output_dir = Path(r"C:\Users\lenovo\Desktop\nnUNet\nnUNet_raw\Dataset001_AISDmini")  # nnUNet 输出目录

    # 指定要处理的病例 ID（手动指定 3 个或全部）
    case_ids = ["0091440", "0091519"]  # ← 修改为你想要的 3 个
    # 或自动检测所有：case_ids = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]

    # =======================

    print(f"将处理 {len(case_ids)} 个病例的 CT 图像")
    print(f"输出目录: {output_dir / 'imagesTr'}")
    print("-" * 60)

    # 转换所有病例
    spacing_info = {}
    for case_id in tqdm(case_ids, desc="转换进度"):
        info = convert_single_case(case_id, base_dir, output_dir)
        if info:
            spacing_info[case_id] = info

    # 保存 spacing 信息（供 mask 转换使用）
    info_path = output_dir / "spacing_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(spacing_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"✓ 完成！成功转换 {len(spacing_info)} 个病例")
    print(f"✓ 图像保存在: {output_dir / 'imagesTr'}")
    print(f"✓ Spacing 信息已保存至: {info_path}")
    print(f"\n下一步: 运行 convert_png_mask_to_nifti.py 转换 mask")


if __name__ == "__main__":
    main()