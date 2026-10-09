# 木材年轮与缺陷多任务项目：论文级数据工程报告

报告日期：2026-08-06

## 1. 结论

最终统一清单包含 **36,747** 张图像：主监督数据 **21,346** 张，辅助数据 **15,401** 张。清单质量状态和派生产物完整性状态均为 **ok**。

源数据只读验证通过：处理前后元数据指纹均为 `31ca6e3d670e8e18ad843677a096bb8a331aead36a98827c49d1a0e102129ad3`，变化数据集 0，关键标注内容变化 0，缺失文件 0。所有新增和修改只发生在 `wood-multitask-final`。

当前数据工程已满足年轮结构、缺陷实例、分类裁剪和双标注桥接四类监督的统一读取要求。模型尚未在本工程上完成正式训练，因此本报告不声称年轮与缺陷已经呈现双向正迁移；第 10 节给出可证伪的论文实验协议。

## 2. 标签与监督策略

- 缺陷检测仅保留 `knot` 和独立 `crack` 两类；检测头含背景时输出通道为 3，YOLO 类别编号为 knot=0、crack=1。
- `knot_with_crack` 只映射为 `knot`，亚型记录为 `cracked`，绝不进入 crack 真值。
- Marrow、overgrown、Quartzity、Blue_stain、resin 不作为目标类别。VSB 中这些标注被排除；OULU/Mokume 的非目标区域保留为 ignore，不被解释为正常背景。
- 分类任务为 `normal/knot/crack`。normal 仅来自已核验负样本；标签缺失始终通过 task mask 隔离。
- Mokume 保留矩形标注，UruDendro 保留真实多边形。年轮标注没有因缺陷矩形化而改动。

原始项目文档中的 VSB 20,276 张和 6 类缺陷统计属于旧方案。本版本以已确认的 VSB 16,000 张、两类监督策略为准，并重新生成全部统计。

## 3. 数据集登记与角色

| 数据集 | 协议角色 | 图像 | 物理组 | 划分 | 年轮 | 缺陷已标注 | 目标实例 | 忽略实例 |
|---|---|---|---|---|---|---|---|---|
| urudendro | primary ring and bridge | 64 | 14 | test=15, train=38, val=11 | 64 | 64 | 802 | 0 |
| urudendro2 | primary ring | 54 | 18 | train=42, val=12 | 54 | 0 | 0 | 0 |
| urudendro4 | external ring test and five-fold R2 | 102 | 21 | external_test=102 | 102 | 0 | 0 | 0 |
| mokume | primary ring and bridge | 1140 | 190 | test=186, train=774, val=180 | 1140 | 258 | 167 | 1 |
| vsb | primary defect train/validation/test | 16000 | 10 | test=4000, train=10000, val=2000 | 0 | 16000 | 30474 | 4574 |
| vnwoodknot | external defect test | 1515 | 1515 | external_test=1515 | 0 | 1515 | 1021 | 0 |
| indiana | external ring test | 1632 | 136 | external_test=1632 | 1466 | 0 | 0 | 0 |
| oulu | external defect test | 839 | 839 | external_test=839 | 0 | 839 | 1431 | 673 |
| wvtec_wood | auxiliary external anomaly evaluation only | 326 | 326 | aux_train_reference=247, external_anomaly_test=79 | 0 | 0 | 0 | 0 |
| coating | auxiliary stress test only | 15075 | 335 | auxiliary_stress=15075 | 0 | 0 | 0 | 0 |
| miscs_code | excluded | 0 | 0 |  | 0 | 0 | 0 | 0 |

WVTec Wood 仅用于外部异常评估；涂层数据仅用于合成压力测试。MiSCS 当前只有代码仓库，没有图像数据，故不计入清单。`mokume-labelme`、`vsb-统计`、`vsb-final.build-cache` 等派生目录不作为独立数据集。除 WVTec 本地记录的许可外，其余数据集在公开发布前仍需逐项确认许可证。

## 4. 物理组划分与防泄漏

- 固定随机种子为 `20260729`，先按物理组划分，再生成目标、COCO 和裁剪。
- 桥接集共 322 张：train=194、val=59、test=69；树或立方体在三个划分间交集为 0。
- VSB 继承 `vsb-final` 的固定采集组划分：train=10,000、val=2,000、test=4,000。
- UruDendro4 锁定为年轮 external_test，并建立 5 折树级 R2；Indiana、VNWoodKnot、OULU 均锁定 external_test。
- 清单重复路径 0、缺失引用 0、不可读图像 0、跨划分物理组泄漏 0。

## 5. 年轮监督产物

已生成 **2,826/2,826** 份年轮目标，错误 0，共 14,130 个 PNG。每张包含 1 px 骨架、5 px 边界、tau=16.0 截断距离场、实例图和有效区。数据集分布：{"indiana": 1466, "mokume": 1140, "urudendro": 64, "urudendro2": 54, "urudendro4": 102}。

共解析 4,319 条闭合轮廓。发现 3 个非严格嵌套警告，样本为：urudendro:F02a, urudendro:F07a, urudendro4:T2_B2_N16_A。这些目标可用于一般训练，但拓扑敏感实验前必须复核原始多边形。

## 6. 两类缺陷 COCO

扩展 COCO 包含 **18,676** 张图像、**33,895** 个目标实例和 **674** 个非目标忽略区域。分割可用实例 31,443 个。

| 数据集 | knot | crack | 目标合计 | 已核验 normal |
|---|---|---|---|---|
| vsb | 28305 | 2169 | 30474 | 1621 |
| vnwoodknot | 1021 | 0 | 1021 | 500 |
| oulu | 1196 | 235 | 1431 | 103 |
| urudendro | 0 | 802 | 802 | 0 |
| mokume | 79 | 88 | 167 | 169 |
| 总计 | 30601 | 3294 | 33895 | 2393 |

按协议划分的目标实例：

| 划分 | knot | crack | 合计 |
|---|---|---|---|
| train | 17722 | 2150 | 19872 |
| val | 3856 | 474 | 4330 |
| test | 6806 | 435 | 7241 |
| external_test | 2217 | 235 | 2452 |

所有框均位于图像范围内，COCO 引用、分割、类别契约和 group/split 继承错误均为 0。VSB 的 16,000 张统一掩膜与 32,095 个裁剪原位复用，没有再复制约 129 GiB 数据。

标准 YOLO 和 Mask R-CNN 导出格式不能表达 ignore 区域：导出器默认使用 box 模式，并保守跳过含 ignore 区域的图像；正式 OULU 全量评估应使用扩展 COCO 加 ignore-aware evaluator，避免把非目标缺陷计作背景。

## 7. 分类裁剪

共索引 **36,288** 个 320×320 裁剪：knot=30,601、crack=3,294、normal=2,393。其中复用 VSB 裁剪 32,095 个，本项目新增 4,193 个。裁剪缺失和物理组/划分继承错误均为 0。

类别明显不平衡，正式训练必须报告 macro-F1、每类召回率和每类 AP，并使用损失重加权或按物理组的平衡采样；不能只报告总体准确率。

## 8. 桥接数据与双任务监督

桥接图像 322 张，划分为 train=194, test=69, val=59；标签状态为 {'verified_negative': 169, 'positive': 152, 'ignore_only': 1}。规范化副本保留 970/970 个形状，其中目标实例 969 个、ignore 实例 1 个。

桥接目标包括 UruDendro crack 802 个、Mokume knot 79 个和 crack 88 个；969 个目标均带 `affects_rings` 属性，可用于缺陷邻域年轮指标和双向条件分支实验。规范化执行实例 ID 去重 6 次、truncated 重算 176 次、自交修复 2 次，并保留 1 个非目标忽略区域。

## 9. 重复与来源完整性审计

主数据 pHash 审计覆盖 21,346 张，阈值为 4：跨划分候选 48 对，完全相同哈希 0 对。VSB 复用 16,000 个哈希，其余新计算 5,346 个。5 张接触表已复核全部 48 对，明显图像重复 0 对；复核为 AI 辅助目视筛查，不是独立领域专家复核，也不能证明物理木材来源必然不同。

WVTec 辅助划分候选 0 对。涂层数据为 335 个基础组，每组 45 个派生、总计 15,075 张，全部处于 auxiliary_stress，不进入主训练/验证结论。

派生产物当前共 19,818 个文件、627.8 MiB，不包含原位复用的 VSB 大文件。关键配置、清单、COCO、裁剪索引和完整性报告的 SHA-256 已写入 `data_processed/provenance/artifact_hashes.json`。

## 10. 双向正迁移验证协议

数据工程只能提供可验证条件，不能单独证明正相关。论文中应将命题表述为“年轮任务与缺陷任务是否产生稳定双向正迁移”，并至少比较以下模型：Ring-only、Defect-only、共享编码器但无交互、R→D Oracle、R→D Pred、D→R Oracle、D→R Pred、完整 BiCRR。

- 年轮指标：Boundary F1/Dice、clDice、ASSD或HD95、闭合率、断裂数；同时报告缺陷邻域和匹配非缺陷区域。
- 缺陷指标：box/mask mAP50-95、AP_knot、AP_crack、每类召回率；分类报告 macro-F1 和 balanced accuracy。
- 统计单位必须是树或立方体等物理组。固定数据划分，至少 5 个训练随机种子，报告均值、标准差、组级 bootstrap 95% CI 和配对置换检验；多重比较使用 Holm 校正。
- 只有完整模型相对无交互共享基线在两个方向都出现正效应、95% CI 不跨 0，并且外部测试不显著退化时，才支持双向正迁移。Oracle 有效而 Pred 无效只能说明条件信号有潜力，不能证明实际模型闭环成立。

详细预注册方案见 `docs/bidirectional_validation_protocol.md`。

## 11. 仍需补齐的论文证据

- 尚无独立第二标注目录，不能报告 10% 双人独立复标、多边形 IoU 或类别一致率。正式论文前必须补做并冻结复核清单。
- Uru 的 3 个非严格嵌套年轮样本需专家复核；在此之前应同时报告包含与排除它们的敏感性分析。
- VNWoodKnot 数字类别 1/2 的生物学含义和所有数据集许可证仍需引用官方来源确认。
- VSB 只有 10 个连续采集组，16,000 张图不等于 16,000 个独立木材来源；置信区间和外推结论必须以采集组为单位解释。
- 数据工程已完成，但尚未生成任何正式模型指标；不得把本报告的完整性 `ok` 写成模型精度结论。

## 12. 复现命令

```powershell
python scripts/build_manifest.py --config configs/data.json --refresh-image-metadata
python scripts/audit_provenance.py --baseline
python scripts/run_data_pipeline.py --stages bridge rings coco crops duplicates --vsb-segmentation-limit 0
python scripts/audit_provenance.py --verify
python scripts/validate_data_artifacts.py
python -m unittest discover -s tests -v
python scripts/build_final_report.py
```

清单 inventory SHA-256：`df49cf1086400375db1d3acd3710893038fa49b1cff120258f17f304b4fae4e3`。关键 manifest SHA-256：`bd364472586a624733c01425fa87ec122fc0c5b31d5a5b583b3db7e7222011f7`。
