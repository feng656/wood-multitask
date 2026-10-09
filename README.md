# 木材年轮与缺陷多任务模型（Wood Ring & Defect Multitask）

共享编码器多任务模型实验仓库：同一个 ResNet34 编码器读取木材横截面 RGB 图像，同时完成
**①年轮提取（边界 + 距离场）②缺陷检测（knot/crack 框 + 掩膜）③缺陷实例分类（normal/knot/crack）**。

> 本仓库是实验全程的产物归档 + 代码。总纲见 [木材年轮与缺陷多任务模型_实验总体整理.md](./木材年轮与缺陷多任务模型_实验总体整理.md)。

## 核心结果（同口径固定 test）

| 模型 | ring Dice | ring MAE | box mAP50-95 | box mAP50 | 分类 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 独立基线（Mokume / YOLOv8s / ResNet34） | 0.3331 | 0.2970 | 0.407 | 0.770 | 0.98585 |
| 旧阶段三（整图 letterbox） | 0.4114 | 0.2463 | 0.2814 | 0.6410 | 0.9709 |
| NO Bridge v1 | 0.4875 | 0.2082 | 0.2588 | 0.6225 | 0.9348 |
| **NO Bridge v2（ep95）** | **0.5172** | **0.1907** | **0.3791** | **0.8941** | **0.9621** |

- 年轮任务显著受益于共享编码器（Dice 超独立基线 **+0.184**）；
- 检测 mAP50 反超 YOLO 基线（0.894 vs 0.770），高 IoU 定位精度（mAP50-95）仍差 0.028；
- 关键阴性结果：「去 bridge 未恢复三任务」证明退化主因是共享编码器本身，而非桥接数据；
- bridge 有无 / BiCRR 有无的消融实验尚未完成（计划见 `06_未完成_后继消融计划/`）。

## 仓库结构

```
├── 木材年轮与缺陷多任务模型_实验总体整理.md   # 实验总纲（结论/方法/数据/全部指标）
├── 00_项目与数据/      # 数据说明、manifest、质量审计、交接文档
├── 01_三条独立基线/    # Mokume 年轮 / YOLOv8s 检测 / ResNet34 分类基线指标
├── 02_旧阶段三_整图letterbox/
├── 03_640原尺度切片版/
├── 04_诊断对照实验/     # 跨域零样本审计、bridge 微调、tile 可学性
├── 05_NO_Bridge共享多任务/   # v1/v2 逐轮指标与配置
├── 06_未完成_后继消融计划/
├── 示例图/
└── code/               # 训练/数据管线代码（scripts + src/wood_data）
```

## 运行

```bash
pip install -e code/
# 训练入口见 code/HANDOFF.md 与 06_未完成_后继消融计划/消融计划与代码入口.md
```

## 未包含内容

- **模型权重**：通过 [GitHub Releases](../../releases) 发布——NO Bridge v2 `best.pt`（最终多任务模型）与 Mokume 年轮基线 `unet_trained_model.pt` 见最新 Release；其余权重（旧阶段三、切片版等，共约 2.5 GB）未入库；
- **数据集本体**（VSB 130 GB / Mokume 26 GB / 各切片与裁剪集）：均为公开数据集，仓库不附带，获取链接见上方「数据来源与参考文献」。

## 数据来源与参考文献

全部实验均使用**公开数据集**完成（权重亦由公开数据集训练），来源如下：

| 用途 | 数据集 | 公开方式 | 参考文献 |
|---|---|---|---|
| 年轮主数据 | UruDendro 系列（UruDendro / UruDendro2 / UruDendro4） | Zenodo，CC BY-NC-SA 4.0 | [1] [2] |
| 年轮主数据 | Mokume | — | — |
| 缺陷主数据 | VSB-TUO（Vysoká Škola Bánská – TU Ostrava） | Zenodo | [3] |
| 缺陷主数据 | VNWoodKnot | Mendeley Data，CC BY 4.0 | [4] |
| 双标注桥接数据 | 补标部分 UruDendro、Mokume 及自建六面样本 | 自建标注 | — |
| 外部验证 | Indiana Hardwood | GitHub 公开 | [5] |
| 外部验证 | OULU-DET | 官网公开 | [6] |

**参考文献**

[1] Marichal, H., Passarella, D., Lucas, P., Profumo, L., Casaravilla, G., Rocha Galli, A., Ambite, F., Randall, G. *UruDendro, a public dataset of 64 cross-section images and manual annual ring delineations of Pinus taeda L.* Annals of Forest Science 82, 25 (2025). https://doi.org/10.1186/s13595-025-01296-5（数据集：https://doi.org/10.5281/zenodo.15110647）

[2] Marichal, H., Blanco, J., Passarella, D., Randall, G. *UruDendro4: A Benchmark Dataset for Automatic Tree-Ring Detection in Cross-Section Images of Pinus taeda L.* 2025 15th IEEE International Conference on Pattern Recognition Systems (ICPRS), pp. 1–7. https://doi.org/10.1109/ICPRS66293.2025.11302831（数据集：https://doi.org/10.5281/zenodo.15653340，CC BY-NC-SA 4.0）

[3] Kodytek, P., Bodzas, A., Bilik, P. *A large-scale image dataset of wood surface defects for automated vision-based quality control processes.* F1000Research 10:581 (2021). 数据：https://doi.org/10.5281/zenodo.4694695

[4] Tran, V., Lam, D., Le, T. *VNWoodKnot: A benchmark image dataset for wood knot detection and classification.* Data in Brief 62:112039 (2025). https://doi.org/10.1016/j.dib.2025.112039

[5] Wu, F., Huang, Y., Benes, B., Warner, C. C., Gazo, R. *Automated tree ring detection of common Indiana hardwood species through deep learning: Introducing a new dataset of annotated images.* Information Processing in Agriculture 11(4):552–558 (2024). https://doi.org/10.1016/j.inpa.2023.10.002（数据/代码：https://github.com/wufanyou/growth-ring-detection）

[6] Luo, Q., Xu, W., Su, J., Yang, C., Gui, W., Silvén, O. *I²GF-Net: Interlayer Information Guidance Feedback Networks for Wood Surface Defect Detection in Complex Texture Backgrounds.* IEEE Transactions on Instrumentation and Measurement (2024). 数据集：http://www.ilove-cv.com/oulu-wood/

## 许可证

代码与文档：[MIT](./LICENSE)。所用公开数据集遵循其各自许可证（UruDendro4 为 CC BY-NC-SA 4.0，VNWoodKnot 为 CC BY 4.0 等）。
