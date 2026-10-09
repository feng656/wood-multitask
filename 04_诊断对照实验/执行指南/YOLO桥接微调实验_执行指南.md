# YOLOv8 桥接微调实验执行指南（单任务对照实验）

目的：回答「bridge 检测差到底是多任务模型的问题，还是数据域/标签本身难学」。
方法：单任务 YOLOv8s 从 VSB 基线权重 `yolov8s_vsb_box-3/weights/best.pt` 出发，
在 **bridge train（194 张，156 Mokume + 38 UruDendro）+ 2000 张 VSB train（seed=42 随机抽样）** 上微调 50 轮，
再用修正过 rectangle 读取的审计脚本在 **bridge test（69 张）** 上分域评估（整图 640 + 640 tile 两种模式）。

## 固定实验参数（已写死在脚本默认值里）

| 参数 | 值 | 说明 |
|---|---|---|
| 训练集 | 194 bridge + 2000 VSB | VSB 抽样 seed=42，可复现 |
| 训练期 val | bridge val 59 张 | 只用于监控/选点，**不是 test** |
| 起始权重 | yolov8s_vsb_box-3/best.pt | 交接文档登记的 VSB 独立基线 |
| epochs / batch / imgsz | 50 / 16 / 640 | 与基线一致 |
| lr0 | 0.005 | 微调用较低学习率，保护 VSB 特征 |
| seed / deterministic | 0 / True | 与基线一致 |

---

## 0. 服务器前置检查

```bash
ssh jzf@10.87.1.64
conda activate py311
cd /home/jzf/dataset-clean/wood-multitask-final
nvidia-smi   # 确认显存。当前 GPU 0 被 speedbatch_v3 占用（约 8.4GB），本实验全部用 GPU 1
```

## 1. 上传 3 个脚本（在本地 Windows 终端执行，sftp 方式）

```bash
sftp jzf@10.87.1.64
# 进入 sftp 后逐条执行：
cd /home/jzf/dataset-clean/wood-multitask-final/scripts
put "E:/dataset-clean/wood-multitask-final/scripts/prepare_yolo_bridge_vsb_finetune.py"
put "E:/dataset-clean/wood-multitask-final/scripts/train_yolo_bridge_vsb_finetune.py"
put "D:/教务处实习/wood_preproject/scripts/audit_yolo_bridge_scale_domain.py"
bye
```

## 2. 生成数据集（服务器）

```bash
python scripts/prepare_yolo_bridge_vsb_finetune.py
```

**预期输出**（conversion_summary.json 里核对）：
- train: **2194** 张（156 mokume + 38 urudendro + 2000 vsb）
- val: **59** 张 bridge；test: **69** 张 bridge
- train 实例 4505（knot 3669 / crack 836），空标注图 370 张（正常，包含背景图）

输出目录：`outputs/defect_baseline/bridge_vsb2000_yolo_box/`（新增，不覆盖任何旧数据）

## 3. 训练（tmux 里跑，防 SSH 断开）

```bash
tmux new -s yolo_ft
python scripts/train_yolo_bridge_vsb_finetune.py --device 1
# 查看：tmux attach -t yolo_ft ；脱离：Ctrl-b 然后 d
```

- 结果目录：`outputs/defect_baseline/runs/bridge_vsb2000_finetune/`（自动编号，不会覆盖）
- 预计 40～90 分钟。训练结束前**不要**对 bridge test 做任何评估。

## 4. 评估（训练结束后，bridge val + bridge test 各跑一次）

```bash
CKPT=/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/bridge_vsb2000_finetune/weights/best.pt

# bridge test（最终口径）
python scripts/audit_yolo_bridge_scale_domain.py \
  --workspace-root /home/jzf/dataset-clean \
  --checkpoint $CKPT \
  --output /home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/bridge_vsb2000_eval_test \
  --split test --mode both --device 1

# bridge val（用于和阶段三多任务模型的 val 指标对比）
python scripts/audit_yolo_bridge_scale_domain.py \
  --workspace-root /home/jzf/dataset-clean \
  --checkpoint $CKPT \
  --output /home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/bridge_vsb2000_eval_val \
  --split val --mode both --device 1
```

每个目录会生成 `audit_metrics.json`，分域（mokume/urudendro）× 模式（full/tile）报告 AP50-95、AP50、Recall50。

## 5. 回传结果（本地执行）

```bash
scp -r jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/bridge_vsb2000_eval_test "D:/教务处实习/wood_preproject/outputs/"
scp -r jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/bridge_vsb2000_eval_val "D:/教务处实习/wood_preproject/outputs/"
scp jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/bridge_vsb2000_finetune/results.csv "D:/教务处实习/wood_preproject/outputs/bridge_vsb2000_finetune_results.csv"
scp jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/bridge_vsb2000_finetune/args.yaml "D:/教务处实习/wood_preproject/outputs/bridge_vsb2000_finetune_args.yaml"
```

## 6. 结果解读（回传后我会做完整对比）

先把已知参照列出来：

| 模型 | 口径 | UruDendro crack | Mokume crack |
|---|---|---|---|
| VSB-only YOLO 零样本 | bridge test, 整图 640 | AP50 0.0035 / R50 0.092 | AP50 0.0423 / R50 0.429 |
| VSB-only YOLO 零样本 | bridge test, 640 tile | AP50 0.0715 / R50 0.454 | — |
| 阶段三多任务 speedbatch_v3 | bridge val, epoch 20 | crack AP50 0.0182 / R50 0.0755（合并域） |
| 阶段三多任务 tiled640 | bridge val, epoch 55 | crack AP50 0.0392 / R50 0.1462（合并域） |
| **本次单任务微调** | bridge test / val | **待回传** |

判断规则：
- 微调后 UruDendro crack（尤其 tile 模式）**明显高于**零样本 0.0715，且明显高于阶段三 val → 桥接数据在单任务检测架构下可学，**多任务竞争是 bridge 差的主因**
- 微调后仍接近零样本水平 → 194 张 bridge train 不足以教会该检测架构，**数据域/标签问题是主因**，多任务不是主要矛盾
- 介于两者之间 → 数据问题是主因，但多任务训练造成了额外损失

## 可选：VSB 能力回归检查

想看加了 bridge 数据后 VSB 检测有没有退化（顺手验证 2000 张 VSB 够不够）：

```bash
python - <<'EOF'
from ultralytics import YOLO
m = YOLO("/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/bridge_vsb2000_finetune/weights/best.pt")
r = m.val(data="/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/yolo_vsb_box/data.yaml", split="test", imgsz=640, batch=16, device=1)
EOF
```

对照基线：VSB test mAP50-95 0.407 / crack AP50-95 0.339。

## 若结果不明确时的备选实验（先别跑，等分析）

- bridge-only 对照（不加 VSB）：`python scripts/prepare_yolo_bridge_vsb_finetune.py --vsb-count 0 --output outputs/defect_baseline/bridge_only_yolo_box`，再微调评估，可区分「bridge 数据本身可学性」与「VSB 稀释影响」

## 注意事项

1. 不要删除或改动正在跑的 speedbatch_v3 训练与 `new history` 文件。
2. 评估脚本只读原图/标注/权重，产物全部写入新目录。
3. bridge test 只在训练结束后评估一次，不要用它调参或选 checkpoint。
