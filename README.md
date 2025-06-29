# Remote Sensing Archaeology with Machine Learning & Georeferenced Analysis

This project aims to automatically discover potential archaeological sites using machine learning and remote sensing data, with full georeferencing and coordinate transformation analysis. All code, data references, and workflows are open and fully path-configurable for cross-platform reproducibility.

## Project Structure

```
your_project/
├── candidate_discovery.py              # Candidate site discovery & clustering
├── coordinate_analysis.py              # Coordinate analysis & georeferencing
├── create_georeferenced_dataset.py     # Build georeferenced datasets
├── feature_extraction_analysis.py      # Feature engineering & visualization
├── ml_model_training.py                # Machine learning model training
├── real_ml_training.py                 # Training on real remote sensing data
├── simplified_ml_training.py           # Simplified ML workflow
├── satellite_data_acquisition.py       # Remote sensing data acquisition
├── solve_dimension_mismatch.py         # Data dimension fix tools
├── georeferenced_archaeology/          # Georeferenced archaeological datasets
├── satellite_analysis/                 # Satellite data analysis scripts & data
├── ml_results/                         # ML results & outputs
│   ├── real_candidates_info.pkl
│   └── coordinate_analysis_report.txt
├── S2/                                 # S2 (Sentinel-2 etc.) tile data
│   └── tile_xxx_S2.tif
├── config.json                         # Centralized path configuration
├── requirements.txt                    # Python dependencies
└── README.md                          # This documentation
```

## Environment & Path Independence

**All paths (scripts, data, models, outputs, visualizations) are managed centrally via `config.json` at the project root.**

- ✅ Every script loads `config.json` at the top—**never hardcode paths**
- ✅ File/directory reading, writing, saving, visualization, etc., must use the `config` variable
- ✅ **100% cross-platform**: works on Windows, Mac, Linux. Only edit `config.json`, not code
- ✅ If command-line arguments are provided, they override `config.json`

### Example config.json

```json
{
  "models_dir": "./satellite_analysis/trained_models",
  "known_sites_csv": "./georeferenced_archaeology/georeferenced_archaeological_sites.csv",
  "known_features_csv": "./satellite_analysis/archaeological_features.csv",
  "output_dir": "./satellite_analysis",
  "georef_output_dir": "./georeferenced_archaeology",
  "results_dir": "./ml_results",
  "s2_dir": "./S2",
  "site_data_dir": "./satellite_analysis/site_data",
  "candidate_discovery_dir": "./satellite_analysis/candidate_discovery",
  "feature_analysis_dir": "./satellite_analysis/feature_analysis",
  "model_evaluation_dir": "./satellite_analysis/model_evaluation",
  "visualizations_dir": "./satellite_analysis/visualizations",
  "ml_results_dir": "./ml_results",
  "masks_dir": "./maya_real_data/masks",
  "dimension_matched_dir": "./maya_real_data/dimension_matched",
  "real_data_dir": "./maya_real_data",
  "georeferenced_sites_map": "./georeferenced_archaeology/georeferenced_sites_map.png",
  "verification_links_csv": "./georeferenced_archaeology/verification_links.csv"
}
```

### Usage

**At the top of every script:**

```python
import os, json
with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
```

**Always use `config["..."]` for paths. Use `os.path.join` and `os.path.abspath` for sub-paths. Never hardcode.**

## Quick Start

### 1. Install Dependencies

It is recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure all data/model/output folders match `config.json` settings.

### 3. Configure Paths

Just edit `config.json`—**no need to modify code**.

### 4. Run Scripts

**Basic usage:**
```bash
python coordinate_analysis.py
```

**Override paths via CLI:**
```bash
python coordinate_analysis.py --results_dir "/your/path/ml_results" --s2_dir "/your/path/S2"
```

## FAQ

**Q: File not found / path error?**  
A: Check your `config.json` settings.

**Q: Cross-platform compatibility?**  
A: All paths are auto-adapted; no need to edit code.

**Q: How to customize paths?**  
A: Simply edit `config.json` or use command-line arguments.

---

## 中文说明

### 项目概述

本项目用于通过机器学习和遥感数据自动发现考古候选点，并进行地理参考与坐标转换分析。

### 路径与环境无关性说明

**本项目所有脚本、数据、模型、输出、可视化等路径，全部通过工程根目录下的 `config.json` 集中管理。**

- 所有.py脚本在顶部自动加载 `config.json`，所有路径引用都通过 `config` 变量获取
- 所有文件/目录的读写、保存、输出、可视化等操作，均不允许硬编码路径
- 支持Windows、Mac、Linux等任意操作系统，无需修改代码，只需调整 `config.json` 即可
- 如有命令行参数，优先参数，否则用 `config.json`

### 路径管理用法

每个脚本顶部统一加载：

```python
import os, json
with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
```

所有路径引用都用 `config["xxx"]`，并用 `os.path.join`、`os.path.abspath` 拼接子路径，保证跨平台。

### 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **准备数据**
   保证所有数据、模型、输出等目录/文件与 `config.json` 一致。

3. **配置路径**
   只需编辑 `config.json` 即可，无需修改任何脚本源码。

4. **运行脚本**
   ```bash
   python coordinate_analysis.py
   ```
   或用命令行参数覆盖：
   ```bash
   python coordinate_analysis.py --results_dir "/your/path/ml_results" --s2_dir "/your/path/S2"
   ```

### 常见问题

- **路径找不到/文件缺失**：请检查 `config.json` 配置是否正确
- **跨平台问题**：所有路径均自动适配，无需手动修改

---

**Contact:** For issues or collaboration, please contact the maintainer.



