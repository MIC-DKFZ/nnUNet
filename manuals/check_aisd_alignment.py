#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AISD 原始数据对齐检查脚本
功能：在转换 nii.gz 前，先检查 DICOM(Image) 和 PNG(Mask) 是否匹配
输入：本地 DICOM 文件夹 和 PNG 文件夹
输出：匹配报告，列出所有问题病例
"""

import os
import re
import json
from pathlib import Path
from PIL import Image
import pydicom
from tqdm import tqdm


def natural_sort_key(s):
    """自然排序"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def check_dicom_info(dicom_folder: Path):
    """
    读取 DICOM 文件夹信息，不加载全部像素（节省内存）
    返回：{
        'num_slices': 切片数量,
        'rows': 行数(高),
        'cols': 列数(宽),
        'spacing': (x,y,z) if available,
        'file_list': 排序后的文件列表
    }
    """
    if not dicom_folder.exists():
        return None, f"文件夹不存在: {dicom_folder}"

    dcm_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith('.dcm')],
                       key=natural_sort_key)

    if len(dcm_files) == 0:
        return None, "无 DICOM 文件"

    # 只读第一个文件获取元数据（不读像素，快）
    try:
        first_file = dicom_folder / dcm_files[0]
        ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)  # 不加载像素数据，快

        rows = ds.Rows
        cols = ds.Columns

        # 获取 spacing（可能不完整，因为每层可能不同，但通常一致）
        pixel_spacing = getattr(ds, 'PixelSpacing', [None, None])
        slice_thickness = getattr(ds, 'SliceThickness', None)

        info = {
            'num_slices': len(dcm_files),
            'rows': rows,
            'cols': cols,
            'pixel_spacing_xy': pixel_spacing,
            'slice_thickness': slice_thickness,
            'file_list': dcm_files[:5]  # 只存前5个文件名示例
        }
        return info, None

    except Exception as e:
        return None, f"读取 DICOM 失败: {e}"


def check_png_info(png_folder: Path):
    """
    检查 PNG 文件夹信息
    返回：{
        'num_files': 文件数量,
        'width': 宽,
        'height': 高,
        'file_list': 排序后的文件列表
    }
    """
    if not png_folder.exists():
        return None, f"文件夹不存在: {png_folder}"

    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')],
                       key=natural_sort_key)

    if len(png_files) == 0:
        return None, "无 PNG 文件"

    # 读第一张获取尺寸
    try:
        first_img = Image.open(png_folder / png_files[0])
        width, height = first_img.size

        info = {
            'num_files': len(png_files),
            'width': width,
            'height': height,
            'file_list': png_files[:5]
        }
        return info, None

    except Exception as e:
        return None, f"读取 PNG 失败: {e}"


def compare_pair(case_id: str, dicom_base: Path, mask_base: Path):
    """
    对比单个病例的 DICOM 和 PNG
    返回：{
        'matched': bool,
        'issues': [问题列表],
        'details': 详细信息
    }
    """
    dicom_folder = dicom_base / case_id / "CT"
    mask_folder = mask_base / case_id

    result = {
        'case_id': case_id,
        'matched': True,
        'issues': [],
        'dicom_info': None,
        'mask_info': None
    }

    # 1. 检查 DICOM
    dcm_info, dcm_err = check_dicom_info(dicom_folder)
    if dcm_info is None:
        result['matched'] = False
        result['issues'].append(f"DICOM错误: {dcm_err}")
        return result
    result['dicom_info'] = dcm_info

    # 2. 检查 PNG
    png_info, png_err = check_png_info(mask_folder)
    if png_info is None:
        result['matched'] = False
        result['issues'].append(f"PNG错误: {png_err}")
        return result
    result['mask_info'] = png_info

    # 3. 对比检查
    issues = []

    # 3.1 层数/数量检查（最关键！）
    if dcm_info['num_slices'] != png_info['num_files']:
        issues.append(f"层数不匹配: DICOM={dcm_info['num_slices']}, PNG={png_info['num_files']}")
        result['matched'] = False

    # 3.2 尺寸检查（XY平面）
    if dcm_info['rows'] != png_info['height'] or dcm_info['cols'] != png_info['width']:
        issues.append(f"尺寸不匹配: DICOM={dcm_info['rows']}x{dcm_info['cols']}, "
                      f"PNG={png_info['height']}x{png_info['width']}")
        result['matched'] = False

    # 3.3 警告：比例严重不符（DICOM spacing 与 PNG 像素比）
    # 如果 DICOM 是 512x512，PNG 是 256x256，可能是下采样过，可以转，但需记录
    dcm_pixels = dcm_info['rows'] * dcm_info['cols']
    png_pixels = png_info['height'] * png_info['width']
    ratio = png_pixels / dcm_pixels if dcm_pixels > 0 else 0

    if ratio < 0.5:  # PNG 像素数少于 DICOM 的 50%
        issues.append(f"警告: PNG分辨率显著低于DICOM (比例{ratio:.2f})，可能已下采样")

    result['issues'] = issues
    return result


def main():
    # ===== 用户配置路径 =====
    dicom_base = Path(r"C:\Users\lenovo\Desktop\ASID数据集")  # DICOM 根目录（含 0091465/CT/*.dcm）
    mask_base = Path(r"C:\Users\lenovo\Desktop\ASID数据集\mask")  # PNG 根目录（含 0091465/*.png）

    # 指定要检查的病例（或自动检测）
    case_ids = ["0091440", "0091519"]  # 手动指定，或改为自动检测所有

    # =======================

    print(f"开始检查 {len(case_ids)} 个病例的数据对齐...")
    print(f"DICOM路径: {dicom_base}")
    print(f"PNG路径: {mask_base}")
    print("=" * 70)

    results = []
    matched_count = 0

    for case_id in tqdm(case_ids, desc="检查进度"):
        result = compare_pair(case_id, dicom_base, mask_base)
        results.append(result)

        if result['matched']:
            matched_count += 1
            print(f"\n[✓] {case_id}: 完全匹配")
            print(f"    DICOM: {result['dicom_info']['num_slices']}层, "
                  f"{result['dicom_info']['rows']}x{result['dicom_info']['cols']}")
            print(f"    PNG: {result['mask_info']['num_files']}张, "
                  f"{result['mask_info']['height']}x{result['mask_info']['width']}")
        else:
            print(f"\n[✗] {case_id}: 发现问题")
            for issue in result['issues']:
                print(f"    - {issue}")

    # 汇总报告
    print("\n" + "=" * 70)
    print(f"检查完成: {matched_count}/{len(case_ids)} 个病例完全匹配")

    if matched_count < len(case_ids):
        print(f"\n⚠️ 警告: 有 {len(case_ids) - matched_count} 个病例存在问题，建议先修复再转换")
        print("\n建议处理方式:")
        print("1. 层数不匹配: 需要检查 PNG 是否对应 DICOM 的特定层（跳过无病灶层）")
        print("2. 尺寸不匹配: PNG 可能被裁剪，需要还原或调整转换脚本")
        print("3. 参考 AISD 的 info.json 文件（如果有）获取层对应关系")
    else:
        print("\n✅ 所有病例检查通过！可以安全执行转换脚本。")
        print("下一步: 运行 convert_dicom_to_nifti.py 和 convert_png_mask_to_nifti.py")


if __name__ == "__main__":
    main()