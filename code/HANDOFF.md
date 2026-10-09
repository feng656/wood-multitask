# Wood Multitask Handoff

更新日期：2026-08-07

## 1. 当前结论

数据工程层已经完成并通过完整性检查；正式模型训练和论文级指标尚未完成。当前没有在本项目中生成正式 `outputs/runs/checkpoints`，也没有可报告的模型精度。

本交接文档以以下协议为准：

- 检测/实例分割类别：`knot`、`crack`
- 一级分类类别：`normal`、`knot`、`crack`
- 非目标缺陷：`ignore`，不能当作 `normal` 或 `other`
- 物理组先划分、再生成目标和裁剪；固定种子：`20260729`
- 原年轮 JSON 不修改；缺陷 JSON 独立保存，由 manifest/loader 联合读取

## 2. 已完成内容

### 数据清单和完整性

- 统一清单：36,747 张图像；主监督 21,346，辅助 15,401。
- 年轮目标：2,826/2,826 张生成成功，14,130 个 PNG，错误 0。
- 缺陷 COCO：18,676 张图像、33,895 个目标实例、674 个 ignore 区域；目标为 knot=30,601、crack=3,294。
- 分类裁剪：36,288 个；knot=30,601、crack=3,294、normal=2,393。
- 桥接集：322 张，固定 train/val/test=194/59/69；跨物理组划分无泄漏。
- 源数据只读校验通过，源数据指纹未变化，缺失文件、引用错误和不可读图像为 0。

证据文件：

- `data_processed/integrity_report.json`
- `docs/paper_grade_data_engineering_report.md`
- `docs/bidirectional_validation_protocol.md`
- `data_processed/provenance/source_verification.json`

### 人工桥接标注

#### UruDendro

- 规范化缺陷 JSON：`data_processed/bridge_annotations/urudendro/`
- 64/64 张已有缺陷 JSON。
- 当前包含 802 个 crack 多边形。
- 原年轮标注仍独立保存，未被覆盖。

#### Mokume

- 规范化目录：`data_processed/bridge_annotations/mokume/`
- 258 个 JSON = 43 个完整立方体 × A-F 六面。
- 28 个缺陷阳性立方体，15 个六面全空的正常立方体，数量符合原要求。
- 88 个阳性面；目标实例 knot=79、crack=88，共 167 个。
- 169 个已核验空面；另有 1 个 ignore-only 面：`CN09_D` 的 `other|outer_damage...`。
- 原始目录中的 B01 不完整，仅 A 面有单点标签，B-F 缺失，未纳入 43 个完整立方体。

### 重要未完成标注

- 尚无第二位标注者的独立复标目录、抽样表或一致性统计。
- 桥接集 322 张按 10% 计算至少需要独立复标 33 张；应保留独立原始 JSON，并计算类别一致率、实例/多边形 IoU 和漏标差异。
- 发现 3 个年轮非严格嵌套警告：`urudendro:F02a`、`urudendro:F07a`、`urudendro4:T2_B2_N16_A`。拓扑敏感实验前需人工复核，并做包含/排除敏感性分析。

## 3. 已冻结的数据集角色

| 数据集 | 当前角色 |
|---|---|
| VSB | 主缺陷 train/val/test：10,000/2,000/4,000 |
| VNWoodKnot | 外部缺陷测试，1,515 张 |
| OULU | 外部缺陷测试，839 张；含 673 个 ignore 区域 |
| UruDendro | 年轮主数据和桥接，64 张同时有年轮与缺陷监督 |
| Mokume | 年轮主数据和桥接；当前桥接 258 张 |
| UruDendro4 | 当前外部年轮测试；后续 20-30 张缺陷丰富图只作外部联合测试 |
| Indiana | 外部年轮测试 |
| WVTec Wood/涂层 | 辅助异常或压力测试，不进入主训练结论 |

注意：VNWoodKnot 当前是干净外部测试集。主线实验不能先训练 VSB 再把全部 VNWoodKnot 加入训练；如需 `VSB+VNWoodKnot`，必须另开适配/消融实验并重新定义外部测试。

## 4. 模型和代码状态

### 年轮 Ring-only

- 项目中只有年轮目标生成器，没有独立 Ring-only 推理/训练/评估入口。
- `E:/dataset-clean/Mokume/unet_trained_model.pt` 约 138 MB，是旧 Mokume 模型的参数 `state_dict`。
- 仓库内没有与其匹配的模型定义、预处理、输出映射和阈值逻辑。
- 在恢复原模型代码前，不能声称已经复现了 Mokume 年轮基线。

### 缺陷检测基线

当前明确采用 `E:/dataset-clean/yolov8s.pt`，不是 YOLOv8-seg：

- 模型任务：检测 `detect`
- 标注格式：YOLO box
- 训练数据：主线只使用 VSB
- 不报告 Mask Dice/mIoU；本阶段只报告 box mAP、每类 AP 和 Recall

准备脚本：`scripts/prepare_yolo_defect_dataset.py`

训练入口：`scripts/train_defect_baseline.py`

准备数据时必须显式使用：`--source-datasets vsb --label-format box`。

现有评估器在 segment 模式下存在框转换和一对一匹配口径问题；检测基线应优先用官方 Ultralytics `val` 输出 mAP/AP/Recall，或在正式实验前修正自定义 evaluator。

### 分类基线

- 现有入口支持 ResNet18/34、类别均衡采样、Macro-F1、Balanced Accuracy、每类 P/R/F1 和混淆矩阵。
- 尚未实现 EfficientNet、ConvNeXt 和二级 knot 亚型分类入口。
- 正式训练前需检查 `scripts/train_classification_baseline.py` 的裁剪路径解析和 best checkpoint 选择逻辑。
- 一级分类保持三类；`other` 继续作为 ignore。

### Stage 3 多任务

- `src/wood_data/multitask.py` 是语义原型：共享编码器、年轮边界/距离、语义缺陷 mask、全图分类；没有检测框和实例 mask。
- `src/wood_data/instance_multitask.py` 是更接近目标的版本：共享 ResNet-FPN、Mask R-CNN 检测/实例 mask、Ring FPN 年轮头。
- 当前实例版分类池化使用整图 ROI，不是严格的缺陷实例 ROI；正式实验前需明确是否修改。
- 当前训练主要按任务 loader 选择损失，`task_mask` 没有完整进入损失计算；应在正式实验中记录并验证任务掩码。
- Stage 3 必须先不加入 BiCRR，也不进行年轮到缺陷或缺陷到年轮的条件传递。

### Stage 4 Bridge

- 已有 bridge 微调入口和正负样本采样逻辑。
- 只能使用 bridge train/val 进行微调和选参；bridge test=69 冻结后使用一次。
- Stage 3 与 Stage 4 默认 checkpoint 路径目前不一致，正式运行必须显式传入正确的 `--init-checkpoint`。

## 5. 下一步计划

### 阶段 0：实验前冻结

1. 完成并记录至少 33 张独立第二标注者复标。
2. 冻结 manifest、taxonomy、物理组划分、哈希和训练随机种子。
3. 复核 3 个年轮拓扑警告。
4. 确认服务器 Python、PyTorch、torchvision、Ultralytics 和 CUDA 版本。

### 阶段 1：Ring-only

1. 恢复 Mokume 权重对应的模型结构和预处理。
2. 先做冻结权重推理和固定外部评估。
3. 再训练一个可复现的 Ring-only 模型，作为共享多任务模型的公平独立基线。

### 阶段 2：VSB-only YOLOv8s 检测

服务器数据假定位于 `/home/jzf/dataset-clean`，项目位于 `/home/jzf/dataset-clean/wood-multitask-final`。

```bash
cd /home/jzf/dataset-clean/wood-multitask-final

python scripts/prepare_yolo_defect_dataset.py \
  --project-root /home/jzf/dataset-clean/wood-multitask-final \
  --workspace-root /home/jzf/dataset-clean \
  --source-datasets vsb \
  --label-format box \
  --output /home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/yolo_vsb_box

python scripts/train_defect_baseline.py \
  --weights /home/jzf/dataset-clean/yolov8s.pt \
  --data /home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/yolo_vsb_box/data.yaml \
  --name yolov8s_vsb_box \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --workers 4 \
  --device 0
```

建议放在 `tmux` 或 `screen` 中运行。启动日志第一行必须显示 `task=detect`。显存不足时，将 batch 改为 8 或 4。

训练后权重预计位于：

`outputs/defect_baseline/runs/yolov8s_vsb_box/weights/best.pt`

### 阶段 3：分类基线

1. ResNet34 三类分类。
2. 类别均衡采样。
3. 使用 train 选参、val 选最佳模型、test 只评估一次。
4. 对 knot 单独训练亚型分类器。

### 阶段 4：Shared-no-interaction

使用共享 ResNet34/50 和实例版缺陷头，损失规则为：

```text
L = m_ring * L_ring
  + m_defect * (L_box + L_class + L_mask)
  + m_cls * L_classification
```

- 年轮数据：只计算年轮损失。
- 缺陷数据：只计算缺陷损失。
- 分类裁剪：只计算分类损失。
- 桥接数据：同时计算年轮和缺陷损失。

### 阶段 5：桥接、外测和 BiCRR

1. 先比较三个独立基线与 Shared-no-interaction，检查负迁移。
2. 用 bridge train/val 微调。
3. 一次性评估 bridge test。
4. 之后才实现 Oracle、Pred 和完整 BiCRR。

正式论文实验至少使用 5 个随机种子，以树、立方体或采集组为统计单位，报告组级 bootstrap 置信区间、配对置换检验及 Holm 多重比较校正。

## 6. 指标口径注意事项

- YOLOv8s 检测阶段：box mAP50-95、mAP50、AP_knot、AP_crack、每类 Recall。
- 真实掩膜可用的 VSB/UruDendro：再报告 Mask Dice/mIoU。
- Mokume、VNWoodKnot、OULU 主要是框标注，不能把框直接宣传为真实 mask 分割指标。
- OULU 的 ignore 区域必须使用 ignore-aware 评估，不能简单当背景。
- 年轮至少报告 Boundary Dice/F1、clDice、距离误差、ASSD/HD95、闭合率和断裂数。
- 分类至少报告 Macro-F1、Balanced Accuracy、每类 P/R/F1 和混淆矩阵。

## 7. 关键路径

- 项目：`E:/dataset-clean/wood-multitask-final/`
- 数据完整性：`data_processed/integrity_report.json`
- 工程报告：`docs/paper_grade_data_engineering_report.md`
- 实验协议：`docs/bidirectional_validation_protocol.md`
- 统一清单：`data_processed/manifests/manifest.csv`
- 桥接规范化审计：`data_processed/bridge_annotations/normalization_audit.json`
- YOLO 检测权重：`E:/dataset-clean/yolov8s.pt`

本文件记录的是当前交接状态；任何改动数据、标签契约、数据集角色或测试集定义，都必须先更新本文件和实验协议，再开始正式训练。
