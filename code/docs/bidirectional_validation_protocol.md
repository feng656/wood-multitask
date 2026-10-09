# 年轮-缺陷双向正迁移预注册验证协议

## 研究问题

在完全相同的物理组划分和训练预算下，年轮监督是否改善 knot/crack 识别，缺陷监督是否改善年轮结构提取，并且这种改善能否在预测条件而非 Oracle 条件下稳定复现。

## 固定数据协议

- 数据清单、桥接 train/val/test、VSB 固定划分和 external_test 在训练前冻结。
- 所有裁剪、patch、增强和采样均继承源图 group_id；同一物理组绝不跨划分。
- `knot_with_crack` 只作为 knot；crack 仅使用独立 Crack 或其他数据集明确的 standalone crack/split 标注。
- 非目标缺陷区域进入 ignore，不计作背景；缺失任务标签由 task mask 隔离。

## 模型矩阵

1. Ring-only：独立年轮基线。
2. Defect-only：独立两类检测/分割基线。
3. Shared-no-interaction：共享编码器但任务头之间不交换条件信息。
4. R-to-D Oracle：缺陷头使用真实年轮条件。
5. R-to-D Pred：缺陷头使用模型预测年轮条件。
6. D-to-R Oracle：年轮头使用真实缺陷条件。
7. D-to-R Pred：年轮头使用模型预测缺陷条件。
8. BiCRR：双向预测条件完整模型。

所有模型使用相同骨干、输入尺寸、增强、优化器、训练步数和早停规则。至少使用 5 个训练随机种子，保存每个种子的完整配置、权重哈希和逐物理组预测。

## 指标

- 年轮：Boundary F1/Dice、clDice、ASSD或HD95、闭合率、每图断裂数。
- 缺陷：box/mask mAP50-95、AP50、AP_knot、AP_crack、每类召回率。
- 分类：macro-F1、balanced accuracy、每类 precision/recall/F1。
- 局部分析：缺陷框外扩固定比例后的年轮指标，与同图面积匹配的非缺陷区域比较。

## 统计检验

- 以树、立方体或采集组为重采样单位，执行组级 bootstrap 95% CI。
- 对相同物理组上的模型差值做双侧配对置换检验；多重比较使用 Holm 校正。
- 同时报告绝对差值、相对差值和标准化效应量，不只报告 p 值。
- bridge test=69 只在方案冻结后使用一次；超参数只依据 train/val。

## 判定规则

只有 BiCRR 相对 Shared-no-interaction 在年轮和缺陷两个方向的主要指标均为正、95% CI 不跨 0，且 UruDendro4、Indiana、VNWoodKnot、OULU 外测无预设阈值以上退化时，才支持“双向正迁移”。Oracle 有效但 Pred 无效，结论只能是条件信息具有潜力。任何一侧无改善或外测显著退化，都不能写成稳定双向正相关。

## 敏感性分析

- 包含/排除 3 个非严格嵌套 Uru 样本。
- 按 knot/crack、图像来源、缺陷尺寸、truncated、年轮密度分层。
- 对 crack 使用独立 Crack 真值，单独验证 `knot_with_crack` 从 crack 真值排除后的稳定性。
- 报告 VSB 10 个采集组逐组结果，避免由图像数量掩盖来源偏差。
