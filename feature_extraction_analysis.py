#!/usr/bin/env python3
"""
真实遗址特征提取与分析
基于已获取的卫星数据进行深度特征分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from datetime import datetime
import argparse
import sys
import json

with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_site_data(site_data_dir):
    """加载所有遗址的卫星数据"""

    print("=== 加载遗址卫星数据 ===")

    site_files = [f for f in os.listdir(site_data_dir) if f.endswith('.npz')]
    all_sites_data = {}

    for site_file in site_files:
        site_path = os.path.join(site_data_dir, site_file)
        data = np.load(site_path, allow_pickle=True)

        site_name = data['metadata'].item()['site_name']
        print(f"加载遗址: {site_name}")

        all_sites_data[site_name] = {
            'metadata': data['metadata'].item(),
            'coordinates_lat': data['coordinates_lat'],
            'coordinates_lon': data['coordinates_lon'],
            'bands': {},
            'indices': {}
        }

        # 加载波段数据
        for key in data.keys():
            if key.startswith('band_'):
                band_name = key.replace('band_', '')
                all_sites_data[site_name]['bands'][band_name] = data[key]
            elif key.startswith('index_'):
                index_name = key.replace('index_', '')
                all_sites_data[site_name]['indices'][index_name] = data[key]

    print(f"成功加载 {len(all_sites_data)} 个遗址的数据")
    return all_sites_data

def extract_archaeological_features(sites_data):
    """提取考古特征"""

    print("\n=== 提取考古特征 ===")

    features_list = []

    for site_name, site_data in sites_data.items():
        print(f"分析遗址: {site_name}")

        # 获取遗址中心区域的特征（中心25%区域）
        center_size = 25  # 中心区域大小
        grid_size = site_data['coordinates_lat'].shape[0]
        start_idx = (grid_size - center_size) // 2
        end_idx = start_idx + center_size

        center_region = slice(start_idx, end_idx), slice(start_idx, end_idx)

        # 基础光谱特征
        spectral_features = {}
        for band_name, band_data in site_data['bands'].items():
            center_data = band_data[center_region]
            spectral_features.update({
                f'{band_name}_mean': np.mean(center_data),
                f'{band_name}_std': np.std(center_data),
                f'{band_name}_max': np.max(center_data),
                f'{band_name}_min': np.min(center_data)
            })

        # 光谱指数特征
        index_features = {}
        for index_name, index_data in site_data['indices'].items():
            center_data = index_data[center_region]
            index_features.update({
                f'{index_name}_mean': np.mean(center_data),
                f'{index_name}_std': np.std(center_data),
                f'{index_name}_max': np.max(center_data),
                f'{index_name}_min': np.min(center_data)
            })

        # 纹理特征（基于灰度共生矩阵的简化版本）
        texture_features = calculate_texture_features(site_data['bands']['B08'][center_region])

        # 形状特征（基于考古敏感指数）
        shape_features = calculate_shape_features(site_data['indices']['ARCH'][center_region])

        # 对比度特征（中心vs周边）
        contrast_features = calculate_contrast_features(site_data, center_region)

        # 组合所有特征
        all_features = {
            'site_name': site_name,
            'latitude': site_data['metadata']['center_lat'],
            'longitude': site_data['metadata']['center_lon'],
            **spectral_features,
            **index_features,
            **texture_features,
            **shape_features,
            **contrast_features
        }

        features_list.append(all_features)

    features_df = pd.DataFrame(features_list)
    print(f"提取了 {len(features_df.columns)-3} 个特征维度")

    return features_df

def calculate_texture_features(image_data):
    """计算纹理特征"""

    # 简化的纹理特征计算
    # 基于图像的统计特性

    # 计算梯度
    grad_x = np.gradient(image_data, axis=1)
    grad_y = np.gradient(image_data, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 计算局部方差
    from scipy import ndimage
    local_variance = ndimage.generic_filter(image_data, np.var, size=3)

    texture_features = {
        'texture_gradient_mean': np.mean(gradient_magnitude),
        'texture_gradient_std': np.std(gradient_magnitude),
        'texture_variance_mean': np.mean(local_variance),
        'texture_variance_std': np.std(local_variance),
        'texture_entropy': calculate_entropy(image_data),
        'texture_contrast': np.std(image_data) / (np.mean(image_data) + 1e-8)
    }

    return texture_features

def calculate_entropy(image_data):
    """计算图像熵"""

    # 将图像数据量化为256个级别
    quantized = (image_data * 255).astype(int)
    quantized = np.clip(quantized, 0, 255)

    # 计算直方图
    hist, _ = np.histogram(quantized, bins=256, range=(0, 256))
    hist = hist / np.sum(hist)  # 归一化

    # 计算熵
    entropy = -np.sum(hist * np.log2(hist + 1e-8))

    return entropy

def calculate_shape_features(arch_index):
    """计算形状特征"""

    # 基于考古敏感指数的形状分析
    threshold = np.mean(arch_index) + np.std(arch_index)
    binary_mask = arch_index > threshold

    # 连通组件分析
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(binary_mask)

    shape_features = {
        'shape_num_objects': num_features,
        'shape_total_area': np.sum(binary_mask),
        'shape_area_ratio': np.sum(binary_mask) / binary_mask.size,
        'shape_compactness': calculate_compactness(binary_mask),
        'shape_elongation': calculate_elongation(binary_mask)
    }

    return shape_features

def calculate_compactness(binary_mask):
    """计算紧致度"""

    if np.sum(binary_mask) == 0:
        return 0

    # 计算周长和面积
    from scipy import ndimage
    perimeter = np.sum(ndimage.binary_erosion(binary_mask) != binary_mask)
    area = np.sum(binary_mask)

    # 紧致度 = 4π * 面积 / 周长²
    if perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter**2)
    else:
        compactness = 0

    return compactness

def calculate_elongation(binary_mask):
    """计算伸长度"""

    if np.sum(binary_mask) == 0:
        return 0

    # 找到对象的坐标
    coords = np.where(binary_mask)
    if len(coords[0]) < 2:
        return 0

    # 计算主轴长度比
    coords_array = np.column_stack(coords)
    cov_matrix = np.cov(coords_array.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列

    if eigenvalues[1] > 0:
        elongation = eigenvalues[0] / eigenvalues[1]
    else:
        elongation = 1

    return elongation

def calculate_contrast_features(site_data, center_region):
    """计算中心与周边的对比度特征"""

    contrast_features = {}

    # 获取中心和周边区域
    full_size = site_data['coordinates_lat'].shape[0]
    center_mask = np.zeros((full_size, full_size), dtype=bool)
    center_mask[center_region] = True
    peripheral_mask = ~center_mask

    # 对每个指数计算对比度
    for index_name, index_data in site_data['indices'].items():
        center_mean = np.mean(index_data[center_mask])
        peripheral_mean = np.mean(index_data[peripheral_mask])

        contrast = abs(center_mean - peripheral_mean)
        contrast_features[f'contrast_{index_name}'] = contrast

    return contrast_features

def perform_feature_analysis(features_df, output_dir):
    """执行特征分析"""

    print("\n=== 执行特征分析 ===")

    # 创建分析输出目录
    analysis_dir = os.path.join(output_dir, "feature_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 准备数值特征（排除非数值列）
    numeric_features = features_df.select_dtypes(include=[np.number])
    feature_names = numeric_features.columns.tolist()

    print(f"分析 {len(feature_names)} 个数值特征")

    # 1. 特征统计分析
    feature_stats = numeric_features.describe()
    stats_path = os.path.join(analysis_dir, "feature_statistics.csv")
    feature_stats.to_csv(stats_path)
    print(f"已保存特征统计: {stats_path}")

    # 2. 特征相关性分析
    correlation_matrix = numeric_features.corr()

    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('特征相关性矩阵', fontsize=16, fontweight='bold')
    plt.tight_layout()

    corr_path = os.path.join(analysis_dir, "feature_correlation_matrix.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存相关性矩阵: {corr_path}")

    # 3. 主成分分析
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    pca = PCA()
    pca_features = pca.fit_transform(scaled_features)

    # PCA解释方差图
    plt.figure(figsize=(12, 8))
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.subplot(2, 1, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('主成分')
    plt.ylabel('解释方差比')
    plt.title('主成分解释方差')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95%方差')
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差比')
    plt.title('累积解释方差')
    plt.legend()

    plt.tight_layout()
    pca_path = os.path.join(analysis_dir, "pca_analysis.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存PCA分析: {pca_path}")

    # 4. 遗址聚类分析
    optimal_k = find_optimal_clusters(scaled_features, analysis_dir)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # 在PCA空间中可视化聚类
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('遗址聚类分析 (PCA空间)')
    plt.colorbar(scatter)

    # 添加遗址名称标注
    for i, site_name in enumerate(features_df['site_name']):
        plt.annotate(site_name, (pca_features[i, 0], pca_features[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.subplot(1, 2, 2)
    # 按数据源着色
    sources = features_df['site_name'].apply(lambda x: 'Maya' if 'Maya' in x else 'Ancient')
    source_colors = {'Maya': 'red', 'Ancient': 'blue'}
    colors = [source_colors[source] for source in sources]

    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=colors, alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('遗址分布 (按数据源)')

    # 添加图例
    for source, color in source_colors.items():
        plt.scatter([], [], c=color, label=source, alpha=0.7)
    plt.legend()

    plt.tight_layout()
    cluster_path = os.path.join(analysis_dir, "site_clustering_analysis.png")
    plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存聚类分析: {cluster_path}")

    # 5. 特征重要性分析
    feature_importance = analyze_feature_importance(numeric_features, analysis_dir)

    # 保存分析结果
    analysis_results = {
        'pca_components': pca.components_,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'cluster_labels': cluster_labels,
        'feature_importance': feature_importance,
        'scaler_params': {'mean': scaler.mean_, 'scale': scaler.scale_}
    }

    results_path = os.path.join(analysis_dir, "analysis_results.npz")
    np.savez_compressed(results_path, **analysis_results)
    print(f"已保存分析结果: {results_path}")

    return analysis_results, scaled_features

def find_optimal_clusters(scaled_features, analysis_dir):
    """寻找最优聚类数量"""

    max_k = min(10, len(scaled_features) - 1)
    inertias = []
    silhouette_scores = []

    from sklearn.metrics import silhouette_score

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, labels))

    # 绘制肘部法则和轮廓系数
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertias, 'bo-')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('簇内平方和')
    plt.title('肘部法则')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数分析')

    plt.tight_layout()
    elbow_path = os.path.join(analysis_dir, "optimal_clusters_analysis.png")
    plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 选择轮廓系数最高的k值
    optimal_k = range(2, max_k + 1)[np.argmax(silhouette_scores)]
    print(f"最优聚类数量: {optimal_k}")

    return optimal_k

def analyze_feature_importance(numeric_features, analysis_dir):
    """分析特征重要性"""

    from sklearn.ensemble import RandomForestRegressor

    # 使用随机森林分析特征重要性
    # 这里我们创建一个综合目标变量（基于多个指数的组合）
    target = (numeric_features['NDVI_mean'] +
              numeric_features['ARCH_mean'] +
              numeric_features['contrast_NDVI'])

    features_for_importance = numeric_features.drop(['NDVI_mean', 'ARCH_mean', 'contrast_NDVI'],
                                                   axis=1, errors='ignore')

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features_for_importance, target)

    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': features_for_importance.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title('Top 20 特征重要性分析')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    importance_path = os.path.join(analysis_dir, "feature_importance.png")
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存特征重要性
    importance_csv_path = os.path.join(analysis_dir, "feature_importance.csv")
    feature_importance.to_csv(importance_csv_path, index=False)
    print(f"已保存特征重要性: {importance_csv_path}")

    return feature_importance

def create_feature_analysis_report(features_df, analysis_results, output_dir):
    """创建特征分析报告"""

    report_content = f"""# 真实遗址特征提取与分析报告

## 项目概述
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **分析遗址数量**: {len(features_df)}个
- **提取特征数量**: {len(features_df.select_dtypes(include=[np.number]).columns)}个
- **数据来源**: 基于精确GPS坐标的卫星数据

## 遗址基本信息

"""
    for idx, site in features_df.iterrows():
        report_content += f"""
### {site['site_name']}
- **坐标**: {site['latitude']:.6f}°, {site['longitude']:.6f}°
- **主要光谱特征**:
  - NDVI均值: {site.get('NDVI_mean', 'N/A'):.4f}
  - NDBI均值: {site.get('NDBI_mean', 'N/A'):.4f}
  - 考古指数均值: {site.get('ARCH_mean', 'N/A'):.4f}
"""

    report_content += f"""
## 特征分析结果

### 主成分分析 (PCA)
- **前3个主成分解释方差**: {analysis_results['pca_explained_variance'][:3].sum():.2%}
- **PC1解释方差**: {analysis_results['pca_explained_variance'][0]:.2%}
- **PC2解释方差**: {analysis_results['pca_explained_variance'][1]:.2%}
- **PC3解释方差**: {analysis_results['pca_explained_variance'][2]:.2%}

### 聚类分析
- **识别的遗址群组**: {len(np.unique(analysis_results['cluster_labels']))}个
- **聚类标签**: {analysis_results['cluster_labels'].tolist()}

### 特征重要性 (Top 10)
"""
    top_features = analysis_results['feature_importance'].head(10)
    for idx, row in top_features.iterrows():
        report_content += f"- **{row['feature']}**: {row['importance']:.4f}\n"

    report_content += """
## 考古学意义

### 光谱特征模式
1. **植被胁迫指标 (NDVI)**: 考古遗址通常表现为植被生长异常
2. **土壤亮度异常 (NDBI)**: 地下结构影响土壤反射特性
3. **考古敏感指数 (ARCH)**: 结合多波段信息的考古特征指标

### 遗址类型差异
- **Maya遗址**: 表现出特定的热带环境光谱特征
- **古代遗址**: 在干旱和半干旱环境中的光谱响应

### 特征提取成功指标
- ✅ 成功提取了多维度考古特征
- ✅ 识别了遗址间的光谱差异模式
- ✅ 建立了特征重要性排序
- ✅ 为机器学习建模提供了基础

## 下一步计划
1. 基于提取的特征训练机器学习模型
2. 使用模型在周边区域搜索相似特征
3. 生成具有精确坐标的候选点
4. 进行人工视觉验证
"""

    # 保存报告
    report_path = os.path.join(output_dir, "feature_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"已保存特征分析报告: {report_path}")

def main(site_data_dir, output_dir):
    """主函数"""

    # 加载数据
    sites_data = load_site_data(site_data_dir)

    # 提取特征
    features_df = extract_archaeological_features(sites_data)

    # 保存特征数据
    features_path = os.path.join(output_dir, "archaeological_features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"\n已保存特征数据: {features_path}")

    # 执行特征分析
    analysis_results, scaled_features = perform_feature_analysis(features_df, output_dir)

    # 创建分析报告
    create_feature_analysis_report(features_df, analysis_results, output_dir)

    print(f"\n=== 特征提取与分析完成 ===")
    print(f"特征维度: {len(features_df.columns)-3}")
    print(f"分析遗址: {len(features_df)}")

    return features_df, analysis_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实遗址特征提取与分析")
    parser.add_argument('--site_data_dir', type=str, help='卫星数据目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    args = parser.parse_args()

    # 优先用命令行参数，否则config.json
    if args.site_data_dir and args.output_dir:
        site_data_dir = args.site_data_dir
        output_dir = args.output_dir
    else:
        config = load_config(args.config)
        site_data_dir = config.get('site_data_dir')
        output_dir = config.get('output_dir') or config.get('site_output_dir') or config.get('satellite_analysis_dir')
    if not site_data_dir or not output_dir:
        print("请通过参数或config.json指定 site_data_dir 和 output_dir 路径")
        sys.exit(1)
    features_df, analysis_results = main(site_data_dir, output_dir)
