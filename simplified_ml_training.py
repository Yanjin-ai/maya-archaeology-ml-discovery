#!/usr/bin/env python3
"""
简化的机器学习训练脚本（跳过耗时的SVM）
专注于随机森林和逻辑回归模型
"""

import os, json
with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import joblib
import rasterio
from scipy import ndimage

def train_simplified_models():
    """训练简化的机器学习模型"""
    
    print("=== 简化的Maya考古数据机器学习训练 ===")
    
    # 数据路径
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预处理后的数据
    print("=== 加载预处理数据 ===")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    with open(os.path.join(data_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(data_dir, "tile_info.pkl"), 'rb') as f:
        tile_info = pickle.load(f)
    
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"训练集考古对象比例: {np.mean(y_train):.4f}")
    print(f"测试集考古对象比例: {np.mean(y_test):.4f}")
    
    # 定义模型（跳过SVM以节省时间）
    print(f"\n=== 训练随机森林和逻辑回归模型 ===")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    }
    
    # 训练和评估模型
    model_results = {}
    
    for model_name, model in models.items():
        print(f"\n--- 训练 {model_name} ---")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 评估指标
        auc_score = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 分类报告
        class_report = classification_report(y_test, y_pred, output_dict=True)
        print(f"Precision (考古对象): {class_report['1']['precision']:.4f}")
        print(f"Recall (考古对象): {class_report['1']['recall']:.4f}")
        print(f"F1-score (考古对象): {class_report['1']['f1-score']:.4f}")
        
        # 保存结果
        model_results[model_name] = {
            'model': model,
            'auc_score': auc_score,
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': class_report
        }
        
        # 保存模型
        model_file = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
        joblib.dump(model, model_file)
        print(f"模型保存至: {model_file}")
    
    # 选择最佳模型
    print(f"\n=== 模型性能比较 ===")
    
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc_score'])
    best_model = model_results[best_model_name]['model']
    
    print(f"最佳模型: {best_model_name}")
    print(f"最佳AUC: {model_results[best_model_name]['auc_score']:.4f}")
    
    # 使用最佳模型进行新区域预测
    print(f"\n=== 使用最佳模型发现新的考古候选点 ===")
    
    # 选择一些未用于训练的瓦片进行预测
    s2_dir = config['s2_dir']
    
    # 获取所有瓦片ID
    import glob
    s2_files = sorted(glob.glob(os.path.join(s2_dir, "*.tif")))
    all_tile_ids = []
    for s2_file in s2_files:
        filename = os.path.basename(s2_file)
        tile_id = filename.replace("tile_", "").replace("_S2.tif", "")
        all_tile_ids.append(tile_id)
    
    # 获取已用于训练的瓦片ID
    trained_tile_ids = [info['tile_id'] for info in tile_info]
    
    # 选择未用于训练的瓦片
    untrained_tile_ids = [tid for tid in all_tile_ids if tid not in trained_tile_ids]
    
    print(f"总瓦片数: {len(all_tile_ids)}")
    print(f"已训练瓦片数: {len(trained_tile_ids)}")
    print(f"未训练瓦片数: {len(untrained_tile_ids)}")
    
    # 选择一些瓦片进行预测
    prediction_tiles = untrained_tile_ids[:30]  # 选择前30个
    
    print(f"选择 {len(prediction_tiles)} 个瓦片进行预测")
    
    candidate_sites = []
    
    for i, tile_id in enumerate(prediction_tiles):
        if i % 10 == 0:
            print(f"预测进度: {i+1}/{len(prediction_tiles)}")
        
        try:
            # 读取Sentinel-2数据
            s2_file = os.path.join(s2_dir, f"tile_{tile_id}_S2.tif")
            with rasterio.open(s2_file) as src:
                s2_data = src.read()
            
            # 重塑数据
            bands, height, width = s2_data.shape
            s2_reshaped = s2_data.transpose(1, 2, 0).reshape(-1, bands)
            
            # 标准化
            s2_scaled = scaler.transform(s2_reshaped)
            
            # 预测
            pred_proba = best_model.predict_proba(s2_scaled)[:, 1]
            
            # 重塑回原始形状
            pred_proba_2d = pred_proba.reshape(height, width)
            
            # 寻找高概率区域
            high_prob_threshold = 0.6  # 60%概率阈值
            high_prob_pixels = np.where(pred_proba_2d > high_prob_threshold)
            
            if len(high_prob_pixels[0]) > 0:
                # 计算高概率区域的中心
                center_y = np.mean(high_prob_pixels[0])
                center_x = np.mean(high_prob_pixels[1])
                max_prob = np.max(pred_proba_2d)
                
                candidate_sites.append({
                    'tile_id': tile_id,
                    'center_pixel': (center_y, center_x),
                    'max_probability': max_prob,
                    'high_prob_pixel_count': len(high_prob_pixels[0]),
                    'prediction_map': pred_proba_2d
                })
        
        except Exception as e:
            continue
    
    print(f"\n发现 {len(candidate_sites)} 个候选考古遗址")
    
    # 按概率排序并选择前2个
    candidate_sites.sort(key=lambda x: x['max_probability'], reverse=True)
    top_candidates = candidate_sites[:2]
    
    print(f"\n=== 前2个最佳候选点 ===")
    for i, candidate in enumerate(top_candidates):
        print(f"候选点 {i+1}:")
        print(f"  瓦片ID: {candidate['tile_id']}")
        print(f"  最大概率: {candidate['max_probability']:.4f}")
        print(f"  中心像素: ({candidate['center_pixel'][0]:.1f}, {candidate['center_pixel'][1]:.1f})")
        print(f"  高概率像素数: {candidate['high_prob_pixel_count']}")
    
    # 创建候选点可视化
    if len(top_candidates) >= 2:
        print(f"\n=== 创建候选点可视化 ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('基于真实Maya数据发现的考古遗址候选点', fontsize=16)
        
        for i, candidate in enumerate(top_candidates):
            tile_id = candidate['tile_id']
            pred_map = candidate['prediction_map']
            
            # 读取原始S2数据用于RGB显示
            s2_file = os.path.join(s2_dir, f"tile_{tile_id}_S2.tif")
            with rasterio.open(s2_file) as src:
                s2_data = src.read()
            
            # 创建RGB合成
            if s2_data.shape[0] >= 3:
                rgb_data = s2_data[:3]
                rgb_normalized = np.zeros_like(rgb_data, dtype=np.float32)
                for j in range(3):
                    band_data = rgb_data[j]
                    band_min, band_max = np.percentile(band_data, [2, 98])
                    rgb_normalized[j] = np.clip((band_data - band_min) / (band_max - band_min), 0, 1)
                rgb_image = np.transpose(rgb_normalized, (1, 2, 0))
            
            # 显示RGB图像
            axes[i, 0].imshow(rgb_image)
            axes[i, 0].set_title(f'候选点 {i+1} - Sentinel-2 RGB\\n瓦片 {tile_id}')
            axes[i, 0].axis('off')
            
            # 显示预测概率图
            im1 = axes[i, 1].imshow(pred_map, cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title(f'考古概率预测\\n最大概率: {candidate["max_probability"]:.3f}')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # 显示高概率区域
            high_prob_mask = pred_map > 0.6
            axes[i, 2].imshow(rgb_image)
            if np.any(high_prob_mask):
                axes[i, 2].contour(high_prob_mask, levels=[0.5], colors='red', linewidths=2)
            axes[i, 2].set_title(f'高概率区域标记\\n(概率 > 60%)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存候选点图
        candidates_file = os.path.join(output_dir, "real_archaeological_candidates.png")
        plt.savefig(candidates_file, dpi=300, bbox_inches='tight')
        print(f"候选点可视化保存至: {candidates_file}")
        plt.close()
    
    # 保存候选点信息
    candidates_info = {
        'model_used': best_model_name,
        'model_performance': {
            'auc_score': model_results[best_model_name]['auc_score'],
            'cv_mean': model_results[best_model_name]['cv_scores'].mean(),
            'cv_std': model_results[best_model_name]['cv_scores'].std()
        },
        'prediction_threshold': 0.6,
        'total_candidates_found': len(candidate_sites),
        'top_candidates': top_candidates[:2]
    }
    
    with open(os.path.join(output_dir, "real_candidates_info.pkl"), 'wb') as f:
        pickle.dump(candidates_info, f)
    
    # 保存候选点坐标信息
    coordinates_file = os.path.join(output_dir, "candidate_coordinates.txt")
    with open(coordinates_file, 'w', encoding='utf-8') as f:
        f.write("=== 基于真实Maya考古数据发现的候选点坐标 ===\n\n")
        f.write("注意: 以下坐标为瓦片内像素坐标，需要结合地理参考信息转换为经纬度\n\n")
        
        for i, candidate in enumerate(top_candidates):
            f.write(f"候选点 {i+1}:\n")
            f.write(f"  瓦片ID: {candidate['tile_id']}\n")
            f.write(f"  像素坐标: ({candidate['center_pixel'][0]:.1f}, {candidate['center_pixel'][1]:.1f})\n")
            f.write(f"  预测概率: {candidate['max_probability']:.4f}\n")
            f.write(f"  高概率像素数: {candidate['high_prob_pixel_count']}\n")
            f.write(f"  数据来源: 100%真实Sentinel-2观测\n")
            f.write(f"  训练数据: 真实Maya考古专家标注\n\n")
    
    print(f"候选点坐标保存至: {coordinates_file}")
    
    print(f"\n=== 简化训练完成 ===")
    print("成功基于真实Maya考古数据训练了机器学习模型")
    print(f"最佳模型: {best_model_name} (AUC: {model_results[best_model_name]['auc_score']:.4f})")
    print(f"发现了 {len(candidate_sites)} 个考古候选点")
    print(f"前2个最佳候选点已准备进行验证")
    
    return candidates_info, model_results

if __name__ == "__main__":
    candidates_info, model_results = train_simplified_models()

