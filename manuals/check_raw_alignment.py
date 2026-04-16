#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nnUNet 数据质量验证脚本
检查转换后的 nii.gz 是否符合训练要求
"""

import os
import sys
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Dict, List


class NiftiValidator:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "imagesTr"
        self.labels_dir = self.dataset_dir / "labelsTr"
        self.report = []

    def log(self, msg: str, level: str = "INFO"):
        """记录日志"""
        prefix = {"INFO": "[✓]", "WARN": "[!]", "ERROR": "[✗]"}
        print(f"{prefix.get(level, '[?]')} {msg}")
        self.report.append({"level": level, "msg": msg})

    def check_file_exists(self):
        """检查基本文件结构"""
        if not self.images_dir.exists():
            self.log(f"imagesTr 目录不存在: {self.images_dir}", "ERROR")
            return False

        if not self.labels_dir.exists():
            self.log(f"labelsTr 目录不存在 (如果是推理阶段可忽略)", "WARN")

        img_files = list(self.images_dir.glob("*.nii.gz"))
        if len(img_files) == 0:
            self.log("imagesTr 中没有 .nii.gz 文件", "ERROR")
            return False

        self.log(f"发现 {len(img_files)} 个图像文件")
        return True

    def validate_single_pair(self, img_path: Path, label_path: Path = None) -> Dict:
        """验证单对 Image-Label"""
        results = {"case": img_path.stem.replace("_0000", ""), "passed": True}

        # 1. 读取图像
        try:
            img = sitk.ReadImage(str(img_path))
            img_arr = sitk.GetArrayFromImage(img)  # 注意：sitk 是 (Z,Y,X)，numpy 也是 (Z,Y,X)
        except Exception as e:
            self.log(f"{img_path.name}: 读取失败 - {e}", "ERROR")
            return {**results, "passed": False, "error": "read_failed"}

        # 2. 检查 CT 数据类型和范围
        self._validate_ct_intensity(img, img_arr, results)

        # 3. 检查物理属性 (Spacing, Origin, Direction)
        phys_ok = self._validate_physics(
            img,
            results,
            label=None if label_path is None else sitk.ReadImage(str(label_path))
        )

        # 4. 如果有 Label，检查 Mask
        if label_path and label_path.exists():
            self._validate_mask(label_path, img, results)
        else:
            results["label_exists"] = False
            self.log(f"{results['case']}: 无对应 Label (仅推理)", "INFO")

        return results

    def _validate_ct_intensity(self, img: sitk.Image, arr: np.ndarray, results: Dict):
        """验证 CT 强度属性"""
        # 检查数据类型
        pixel_id = img.GetPixelIDTypeAsString()
        results["dtype"] = pixel_id

        if "float" in pixel_id.lower():
            self.log(f"{results['case']}: CT 为浮点型 {pixel_id} (建议 int16 节省空间)", "WARN")
        elif "int" not in pixel_id.lower():
            self.log(f"{results['case']}: CT 数据类型异常 {pixel_id}", "ERROR")
            results["passed"] = False

        # 检查 HU 值范围 (对于 NCCT)
        min_val, max_val = float(arr.min()), float(arr.max())
        results["intensity_range"] = [min_val, max_val]

        if max_val < 100:  # 如果最大值小于100，可能是归一化后的数据，不是原始 HU
            self.log(f"{results['case']}: CT 最大值仅 {max_val}，可能已归一化丢失 HU 信息", "WARN")
        elif min_val < -3000 or max_val > 3000:
            self.log(f"{results['case']}: CT 值范围异常 [{min_val}, {max_val}]，检查单位", "WARN")
        else:
            self.log(f"{results['case']}: CT 强度范围正常 [{min_val:.0f}, {max_val:.0f}] HU")

    def _validate_physics(self, img: sitk.Image, results: Dict, label: sitk.Image = None) -> bool:
        """验证物理一致性"""
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()
        size = img.GetSize()

        results["spacing"] = list(spacing)
        results["origin"] = list(origin)
        results["size"] = list(size)

        # 检查 Spacing 单位 (应该都是 mm，且为正值)
        for i, s in enumerate(spacing):
            if s <= 0:
                self.log(f"{results['case']}: Spacing[{i}]={s} 为负或零，错误！", "ERROR")
                return False
            if s > 100:  # 如果大于 100，可能是米而不是毫米
                self.log(f"{results['case']}: Spacing[{i}]={s} 过大，单位可能是米而非毫米", "ERROR")
                return False

        # 检查方向矩阵 (应该是单位矩阵或标准方向，不能全0)
        direction_vals = list(direction)
        if all(d == 0 for d in direction_vals):
            self.log(f"{results['case']}: Direction 矩阵全零，坐标系错误！", "ERROR")
            return False

        # 如果有 Label，检查是否完全匹配
        if label is not None:
            l_spacing = label.GetSpacing()
            l_origin = label.GetOrigin()
            l_direction = label.GetDirection()
            l_size = label.GetSize()

            mismatches = []
            if not np.allclose(spacing, l_spacing, rtol=1e-5):
                mismatches.append(f"Spacing: {spacing} vs {l_spacing}")
            if not np.allclose(origin, l_origin, rtol=1e-5):
                mismatches.append(f"Origin: {origin} vs {l_origin}")
            if not np.allclose(direction, l_direction, rtol=1e-5):
                mismatches.append(f"Direction mismatch")
            if size != l_size:
                mismatches.append(f"Size: {size} vs {l_size}")

            if mismatches:
                for m in mismatches:
                    self.log(f"{results['case']}: Image-Label 不匹配 - {m}", "ERROR")
                results["passed"] = False
                return False
            else:
                self.log(f"{results['case']}: 物理属性完全匹配 (Spacing/Origin/Direction)")

        return True

    def _validate_mask(self, label_path: Path, img: sitk.Image, results: Dict):
        """验证 Mask 属性"""
        try:
            label = sitk.ReadImage(str(label_path))
            label_arr = sitk.GetArrayFromImage(label)
        except Exception as e:
            self.log(f"{results['case']}: Label 读取失败 - {e}", "ERROR")
            results["passed"] = False
            return

        # 检查数据类型
        pixel_id = label.GetPixelIDTypeAsString()
        if "unsigned char" not in pixel_id and "short" not in pixel_id and "int" not in pixel_id:
            self.log(f"{results['case']}: Label 数据类型 {pixel_id} 非标准 (建议 uint8)", "WARN")

        # 检查标签值
        unique_vals = np.unique(label_arr)
        results["label_values"] = unique_vals.tolist()

        if len(unique_vals) > 20:  # 如果是连续值而不是离散标签，可能是转换错误
            self.log(f"{results['case']}: Label 值过多 ({len(unique_vals)} 个唯一值)，可能是连续值而非分割标签", "ERROR")
            results["passed"] = False
            return

        # 检查是否有背景 (0)
        if 0 not in unique_vals:
            self.log(f"{results['case']}: Label 中没有 0 (背景)，值域为 {unique_vals}", "WARN")

        # 检查前景占比 (不能全 0，也不能全 1)
        fg_ratio = (label_arr > 0).sum() / label_arr.size
        results["foreground_ratio"] = float(fg_ratio)

        if fg_ratio == 0:
            self.log(f"{results['case']}: Label 全为 0 (无病灶，可能是空标注)", "ERROR")
            results["passed"] = False
        elif fg_ratio > 0.9:
            self.log(f"{results['case']}: Label 前景占比 {fg_ratio:.1%} 过高 (>90%)，可能包含背景误标", "WARN")
        else:
            self.log(f"{results['case']}: Label 值 {unique_vals}，前景占比 {fg_ratio:.2%}")

    def check_all_cases(self, limit: int = None):
        """批量检查所有病例"""
        if not self.check_file_exists():
            return False

        img_files = sorted(self.images_dir.glob("*_0000.nii.gz"))
        if limit:
            img_files = img_files[:limit]

        self.log(f"开始验证 {len(img_files)} 个病例...")

        all_results = []
        for img_path in img_files:
            # 对应 label 文件名：去掉 _0000
            case_name = img_path.name.replace("_0000.nii.gz", "")
            label_path = self.labels_dir / f"{case_name}.nii.gz"

            result = self.validate_single_pair(img_path, label_path)
            all_results.append(result)

        # 统计
        passed = sum(1 for r in all_results if r.get("passed"))
        self.log(f"\n验证完成: {passed}/{len(all_results)} 个通过", "INFO" if passed == len(all_results) else "WARN")

        # 保存报告
        report_path = self.dataset_dir / "validation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {"total": len(all_results), "passed": passed},
                "details": all_results,
                "logs": self.report
            }, f, indent=2, ensure_ascii=False)

        self.log(f"详细报告已保存: {report_path}")
        return all_results


if __name__ == "__main__":
    # 使用示例
    dataset_path = r"C:\Users\lenovo\Desktop\nnUNet\nnUNet_raw\Dataset001_AISDmini"  # 修改为你的路径

    validator = NiftiValidator(dataset_path)
    results = validator.check_all_cases(limit=3)  # 先检查前3个，确认无误后再检查全部