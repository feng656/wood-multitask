# VSB YOLOv8s 到桥接集的受控审计

日期：2026-08-12

## 审计问题

使用同一个 VSB-only YOLOv8s checkpoint，区分以下可能原因：

1. 桥接标签或坐标转换错误；
2. 约 2000-3800 像素原图直接缩放到 640，导致细裂纹丢失；
3. VSB 与 UruDendro/Mokume 的域差异；
4. 跨 tile 合并造成召回下降；
5. 多任务竞争或阶段三 RPN 故障。

Checkpoint：`D:/教务处实习/wood_preproject/outputs/defect_baseline/runs/yolov8s_vsb_box-3/weights/best.pt`

## 已证实的标签问题

旧 `eval_yolo_bridge.py` 只接受不少于 3 个点的形状。Mokume 的 26 个 test 标注全是 LabelMe rectangle（2 个角点），因此旧脚本把它们全部漏掉。修复后 test GT 为：

| 域 | 图像 | knot | crack | 无主类缺陷图像 |
|---|---:|---:|---:|---:|
| Mokume | 54 | 19 | 7 | 36 |
| UruDendro | 15 | 0 | 185 | 0 |

这证明旧合并指标的标签口径有误，但不能据此认为模型本身没有跨域问题。

## 同一权重的整图结果

推理为原图整体送入 YOLO、`imgsz=640`、置信度下限 0.001。低下限用于保留 AP 排序，不等于部署阈值。

| 域/类别 | mAP50-95 | AP50 | Recall@0.5 |
|---|---:|---:|---:|
| Mokume knot | 0.0000 | 0.0000 | 0.0000 |
| Mokume crack | 0.0179 | 0.0423 | 0.4286 |
| UruDendro crack | 0.00085 | 0.00352 | 0.0919 |

UruDendro 是合并 bridge crack 指标极低的主要来源。Mokume crack 不是完全检测不到，但只有 7 个 GT，统计不稳定；Mokume knot 完全未召回。

## 尺度受控对照

UruDendro crack 原图框中位尺寸约 79.4 x 90.8 像素；整图缩到 640 后中位尺寸约 20.2 x 23.1 像素，10/185 个实例至少一边小于 4 像素。

同一权重改用原分辨率 640 tile、160 overlap，并按阶段三的“不同 tile + IoS 0.2”合并：

| UruDendro crack | mAP50-95 | AP50 | Recall@0.5 |
|---|---:|---:|---:|
| 整图 640 | 0.00085 | 0.00352 | 0.0919 |
| 原分辨率 tile | 0.0332 | 0.0715 | 0.4541 |

Recall 提升约 4.94 倍，AP50 提升约 20.3 倍。因此输入尺度是明确且重要的原因。

反面证据：tile 后 AP50 仍只有 0.0715，而且低阈值候选很多。尺度不能单独解释全部失败；VSB 与 UruDendro 的外观/标签定义差异、置信度排序和缺少 bridge 域适配仍存在。

Mokume 的大矩形缺陷经常跨越多个 640 tile。简单把 tile 预测映射回整图后与完整大框做 IoU，会得到 0 recall；这不能直接证明模型没看见局部缺陷。Mokume 必须同时报告 tile-local recall 和整实例重建 recall，或改用能表示整实例关系的评价。

## 对阶段三的含义

1. YOLO 是独立单任务模型，仍然跨域崩溃，所以多任务竞争不是唯一根因。
2. YOLO 没有使用 bridge train，因此这次结果不能证明 bridge 数据不可学，只能证明 VSB-only 零样本迁移失败。
3. 阶段三 val 的支持数 `knot=12, crack=212` 与当前 manifest 的 val 完全一致（Mokume 12 knot + 9 crack，UruDendro 203 crack），不是漏标。YOLO 本次使用 test（19 knot、192 crack），两者不能直接比较绝对 AP。
4. 当前 history 没有 RPN proposal recall，不能把“新 RPN 召回严重不足”当作已证实结论。oracle box 分类 F1 高只证明给定真实框后类别可分，不能区分 RPN、ROI 回归和后处理。
5. bridge train 只有 8 个 UruDendro 物理组，却包含 414 个 crack 实例。独立物理样本少是阶段三过拟合风险，但不是 VSB-only YOLO 失败的直接原因。

## 其他代码/数据风险

- `prepare_bridge_yolo_dataset.py` 遇到任意 ignored shape 时会跳过整图。当前 val 有 1 张 only-other 图，train/test 没有有效主类与 ignored 混合图，因此本版数据未丢主类；未来标注混合时该逻辑应改成仅忽略对应实例/区域。
- Mokume 标注为 rectangle，不是真实缺陷轮廓。其 box/class 监督有效，但把矩形内部全部当 mask 会产生伪掩膜。建议 Mokume 只参与 box/class loss，mask loss 和 mask AP 仅对 true polygon 实例计算。
- 本地 torch/torchvision C++ NMS 不匹配。审计脚本提供了等价纯 PyTorch NMS 后备；服务器已有训练环境不应为此重装依赖。

## 下一步，按证据优先级

1. 在阶段三 checkpoint 上按 UruDendro/Mokume 分域报告 box/mask AP、Recall。
2. 新增 RPN proposal recall@0.3/@0.5，分别 top-100/top-1000，并报告 train/val。
3. 同时记录 tile-local recall、映射后未合并 recall、阶段三实际合并后 recall。
4. 若 proposal recall 高而最终 AP 低，优先修 ROI 回归、得分排序和合并；若 proposal recall 低，再考虑 RPN warm-up/anchor/域均衡采样。
5. 用 bridge train/val 做单任务可学性对照，不使用 test 调参数。只有 bridge-only train/val 均低时，才优先怀疑标签或结构不匹配。
6. 暂不重划 test，先保留为固定对照。若增加物理组，应新建版本化 split，并同时保留旧 split 的结果。

## 产物

- `scripts/audit_yolo_bridge_scale_domain.py`：独立、只读的整图/切片审计脚本。
- `scripts/eval_yolo_bridge.py`：已修复 LabelMe rectangle 漏读。
- `outputs/yolo_bridge_audit/label_audit.json`：标签和尺寸审计。
- `outputs/yolo_bridge_audit_full_final/audit_metrics.json`：修正后的整图结果。
- `outputs/yolo_bridge_audit_tile_urudendro_final/audit_metrics.json`：UruDendro 切片对照。
