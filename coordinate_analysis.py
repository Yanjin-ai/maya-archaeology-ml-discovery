#!/usr/bin/env python3 
"""
分析候选点的真实世界坐标
处理地理参考信息和坐标转换
"""

import os
import numpy as np
import rasterio
import pickle
import argparse
import json
import sys
import glob

with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_real_world_coordinates(results_dir, s2_dir):
    """分析候选点的真实世界坐标"""

    print("=== 候选点真实世界坐标分析 ===")

    try:
        with open(os.path.join(results_dir, "real_candidates_info.pkl"), 'rb') as f:
            candidates_info = pickle.load(f)

        top_candidates = candidates_info['top_candidates']

        print(f"发现的候选点数量: {len(top_candidates)}")

        # 分析每个候选点的地理信息
        coordinate_analysis = []

        for i, candidate in enumerate(top_candidates):
            tile_id = candidate['tile_id']
            pixel_coords = candidate['center_pixel']

            print(f"\n--- 候选点 {i+1} 坐标分析 ---")
            print(f"瓦片ID: {tile_id}")
            print(f"像素坐标: ({pixel_coords[0]:.1f}, {pixel_coords[1]:.1f})")

            # 读取对应的S2文件检查地理参考信息
            s2_file = os.path.join(s2_dir, f"tile_{tile_id}_S2.tif")

            if os.path.exists(s2_file):
                with rasterio.open(s2_file) as src:
                    # 获取地理参考信息
                    transform = src.transform
                    crs = src.crs
                    bounds = src.bounds

                    print(f"坐标系: {crs}")
                    print(f"变换矩阵: {transform}")
                    print(f"边界: {bounds}")

                    # 尝试将像素坐标转换为地理坐标
                    if transform and not transform.is_identity:
                        # 使用仿射变换将像素坐标转换为地理坐标
                        geo_x, geo_y = rasterio.transform.xy(transform, pixel_coords[0], pixel_coords[1])
                        print(f"地理坐标: ({geo_x:.6f}, {geo_y:.6f})")

                        coordinate_analysis.append({
                            'candidate_id': i+1,
                            'tile_id': tile_id,
                            'pixel_coords': pixel_coords,
                            'geo_coords': (geo_x, geo_y),
                            'crs': str(crs),
                            'has_georeference': True
                        })
                    else:
                        print("警告: 该瓦片没有有效的地理参考信息")
                        print("无法转换为真实世界坐标")

                        coordinate_analysis.append({
                            'candidate_id': i+1,
                            'tile_id': tile_id,
                            'pixel_coords': pixel_coords,
                            'geo_coords': None,
                            'crs': str(crs) if crs else None,
                            'has_georeference': False
                        })
            else:
                print(f"错误: 找不到瓦片文件 {s2_file}")

        # 检查数据集的地理参考状况
        print(f"\n=== 数据集地理参考状况检查 ===")

        # 随机检查几个瓦片的地理参考信息
        s2_files = glob.glob(os.path.join(s2_dir, "*.tif"))[:10]  # 检查前10个文件

        georef_status = []

        for s2_file in s2_files:
            filename = os.path.basename(s2_file)
            tile_id = filename.replace("tile_", "").replace("_S2.tif", "")

            with rasterio.open(s2_file) as src:
                transform = src.transform
                crs = src.crs

                has_valid_georeference = (
                    transform and 
                    not transform.is_identity and 
                    crs is not None
                )

                georef_status.append({
                    'tile_id': tile_id,
                    'has_georeference': has_valid_georeference,
                    'crs': str(crs) if crs else None,
                    'transform': str(transform)
                })

        # 统计地理参考状况
        georef_count = sum(1 for status in georef_status if status['has_georeference'])
        total_checked = len(georef_status)

        print(f"检查的瓦片数: {total_checked}")
        print(f"有地理参考的瓦片数: {georef_count}")
        print(f"地理参考比例: {georef_count/total_checked*100:.1f}%")

        if georef_count == 0:
            print("\n⚠️  重要发现: 数据集缺乏地理参考信息")
            print("这意味着:")
            print("1. 像素坐标无法直接转换为经纬度")
            print("2. 候选点位置需要通过其他方式确定")
            print("3. 可能需要查阅原始数据集的元数据")

        # 保存坐标分析结果
        output_file = os.path.join(results_dir, "coordinate_analysis.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump({
                'candidates': coordinate_analysis,
                'georef_status': georef_status,
                'summary': {
                    'total_candidates': len(coordinate_analysis),
                    'georeferenced_candidates': sum(1 for c in coordinate_analysis if c['has_georeference']),
                    'dataset_georef_ratio': georef_count/total_checked if total_checked > 0 else 0
                }
            }, f)

        # 生成坐标报告
        coord_report_file = os.path.join(results_dir, "coordinate_analysis_report.txt")
        with open(coord_report_file, 'w', encoding='utf-8') as f:
            f.write("=== 候选点坐标分析报告 ===\n\n")
            f.write("数据来源: Kokalj et al. (2023) Maya考古数据集\n")
            f.write("研究区域: 墨西哥Chactún古Maya城市中心\n\n")

            f.write("=== 候选点坐标信息 ===\n")
            for analysis in coordinate_analysis:
                f.write(f"\n候选点 {analysis['candidate_id']}:\n")
                f.write(f"  瓦片ID: {analysis['tile_id']}\n")
                f.write(f"  像素坐标: ({analysis['pixel_coords'][0]:.1f}, {analysis['pixel_coords'][1]:.1f})\n")

                if analysis['has_georeference']:
                    f.write(f"  地理坐标: ({analysis['geo_coords'][0]:.6f}, {analysis['geo_coords'][1]:.6f})\n")
                    f.write(f"  坐标系: {analysis['crs']}\n")
                else:
                    f.write(f"  地理坐标: 无法确定（缺乏地理参考）\n")
                    f.write(f"  坐标系: {analysis['crs'] or '未知'}\n")

            f.write(f"\n=== 地理参考状况 ===\n")
            f.write(f"检查瓦片数: {total_checked}\n")
            f.write(f"有地理参考瓦片数: {georef_count}\n")
            f.write(f"地理参考比例: {georef_count/total_checked*100:.1f}%\n")

            if georef_count == 0:
                f.write(f"\n=== 重要说明 ===\n")
                f.write("数据集缺乏有效的地理参考信息，这是学术数据集的常见情况。\n")
                f.write("原因可能包括:\n")
                f.write("1. 数据集为了机器学习优化而移除了地理参考\n")
                f.write("2. 数据经过预处理和标准化\n")
                f.write("3. 保护敏感考古位置信息\n\n")
                f.write("建议解决方案:\n")
                f.write("1. 查阅原始论文的补充材料\n")
                f.write("2. 联系数据集作者获取地理参考信息\n")
                f.write("3. 使用相对位置关系进行分析\n")

        print(f"\n坐标分析报告保存至: {coord_report_file}")

        return coordinate_analysis, georef_status

    except Exception as e:
        print(f"坐标分析失败: {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析候选点的真实世界坐标")
    parser.add_argument('--results_dir', type=str, help='ml结果目录')
    parser.add_argument('--s2_dir', type=str, help='S2数据目录')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    args = parser.parse_args()

    # 优先命令行参数，否则读取config.json
    if args.results_dir and args.s2_dir:
        results_dir = args.results_dir
        s2_dir = args.s2_dir
    else:
        config = load_config(args.config)
        results_dir = config.get('results_dir')
        s2_dir = config.get('s2_dir')

    if not results_dir or not s2_dir:
        print("请通过参数或config.json指定results_dir和s2_dir路径")
        sys.exit(1)

    coord_analysis, georef_status = analyze_real_world_coordinates(results_dir, s2_dir)
