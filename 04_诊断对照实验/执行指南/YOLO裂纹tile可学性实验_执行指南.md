# YOLOv8 裂纹中心 tile 可学性实验执行指南（实验二）

目的：回答「裁剪出来的 640 tile 里裂纹到底能不能看清、裁剪图能不能当训练数据用」。
方法：训练集 = Mokume 全部 259 张标注面 + 裂纹中心 tile（38 张 urudendro train 源图里按**源图名**留 8 张不动）+ 1500 张 VSB；
测试集 = 那 8 张源图的所有 tile（模型训练时从未见过这些源图的任何像素）。
如果模型在"没见过的木板"的 tile 上能检出裂纹 → 裂纹在 640 原生尺度下是可分辨的、裁剪图可用。

## 数据划分（脚本已固定，seed=42）

| 划分 | 内容 | 数量 |
|---|---|---|
| train | 30 张 train 源图的 tile + 259 Mokume + 1500 VSB | 2109 张（crack 6021 / knot 2763 / 空 382） |
| val | 11 张原 val 源图的 tile（训练监控） | 203 张（crack 1706） |
| test | **8 张 holdout train 源图的 tile（最终口径）** | 32 张（crack 71） |
| ext_test | 15 张原 test 源图的 tile（外部复核） | 191 张（crack 1241） |

- holdout 源图（seed=42）：F04a, F07a, F07b, F07c, L02b, L02e, L09a, L09b
- tile 标注 = 源图裂纹框裁剪到 tile 窗口的相交框；三个测试集合的源图**全部**未进训练，无纹理泄漏
- 训练超参：batch 16 / imgsz 640 / lr0 0.005 / seed 0，起点 VSB best.pt；**epochs 500**（比实验一更长，给足模型拟合机会，让"能不能看清"的判定更硬：500 轮饱和训练都学不会，就是真学不会）

---

## 0. 服务器前置检查

```bash
ssh jzf@10.87.1.64
conda activate py311
cd /home/jzf/dataset-clean/wood-multitask-final
nvidia-smi   # 用 GPU 1。如果实验一 bridge_vsb2000_finetune 还在 GPU 1 上跑，等它结束（两个 640/batch16 训练放不下一张卡）
```

## 1. 上传（本地 Windows 终端，sftp）

```bash
sftp jzf@10.87.1.64
# 进入 sftp 后逐条执行：
cd /home/jzf/dataset-clean/wood-multitask-final/scripts
put "E:/dataset-clean/wood-multitask-final/scripts/prepare_yolo_tile_learnability.py"
put "E:/dataset-clean/wood-multitask-final/scripts/train_yolo_bridge_vsb_finetune.py"
# 上传整个裁剪目录（432MB，几分钟）。先建目录再递归 put：
mkdir /home/jzf/dataset-clean/UnuDendro系列/UruDendro/裁剪640
put -r "E:/dataset-clean/UnuDendro系列/UruDendro/裁剪640" /home/jzf/dataset-clean/UnuDendro系列/UruDendro/
bye
```

如果 sftp 版本不支持 `put -r`（会报错），改打包方式：

```bash
# 本地打包
cd /e/dataset-clean/UnuDendro系列/UruDendro && tar -czf /e/dataset-clean/caijian640.tar.gz 裁剪640
# sftp 里上传单个 tar 包，然后服务器上解压：
#   tar -xzf /home/jzf/dataset-clean/caijian640.tar.gz -C /home/jzf/dataset-clean/UnuDendro系列/UruDendro/
```

## 2. 服务器数据完整性核对

```bash
# Mokume 标注面应有 259 个 defects json（训练集里 Mokume 的预期数量）
find /home/jzf/dataset-clean/Mokume/MokumeDataset/MokumeDataset -path '*/defects/*.json' | wc -l
# 裁剪目录应有 776 个 tile + 1 个 tile_manifest.csv
ls /home/jzf/dataset-clean/UnuDendro系列/UruDendro/裁剪640 | wc -l
```

Mokume 数量不是 259 就告诉我，可能要把本地的 MokumeDataset 目录也传上去（本地的完整版有 259 面）。

## 3. 生成数据集（服务器）

```bash
python scripts/prepare_yolo_tile_learnability.py
```

**预期输出**（conversion_summary.json 里核对）：
- train: 2109 张；val: 203；test: 32；ext_test: 191
- train 实例：crack 6021 / knot 2763，空标注 382 张
- holdout_sources = F04a, F07a, F07b, F07c, L02b, L02e, L09a, L09b

输出目录：`outputs/defect_baseline/tile_learnability_yolo/`（新增，不覆盖任何旧数据）

## 4. 训练（tmux 里跑）

```bash
tmux new -s yolo_ft2
python scripts/train_yolo_bridge_vsb_finetune.py \
  --data outputs/defect_baseline/tile_learnability_yolo/data.yaml \
  --name tile_learnability_finetune --device 1 --epochs 500
# 查看：tmux attach -t yolo_ft2 ；脱离：Ctrl-b 然后 d
```

- 结果目录：`outputs/defect_baseline/runs/tile_learnability_finetune/`
- 预计 8～12 小时（2109 张 × 500 轮 ≈ 6.6 万次迭代），挂 tmux 里过夜跑即可。训练结束前**不要**跑下面的评估。
- 如果之前用 50 轮启动过一次，重跑会自动存到 `tile_learnability_finetune2/`，不会覆盖。
- ultralytics 默认 patience=100：val 指标连续 100 轮不涨会提前停，属于正常收敛，不用管。
- best.pt 仍按 val（203 张独立源图 tile）挑选，不会因为轮数多而过拟合到 test。

## 5. 评估（训练结束后，三个独立集合各测一次）

```bash
CKPT=/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/tile_learnability_finetune/weights/best.pt

python - <<'EOF' 2>&1 | tee tile_learnability_eval.log
from ultralytics import YOLO
CKPT = "/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/tile_learnability_finetune/weights/best.pt"
DATA = "/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/tile_learnability_yolo"
m = YOLO(CKPT)
print("===== test: 8 张 holdout 源图的 tile（最终口径）=====")
m.val(data=f"{DATA}/data.yaml", split="test", imgsz=640, batch=16, device=1)
print("===== val: 原 val 源图的 tile =====")
m.val(data=f"{DATA}/data.yaml", split="val", imgsz=640, batch=16, device=1)
print("===== ext_test: 原 test 源图的 tile（外部复核）=====")
m.val(data=f"{DATA}/data_ext_test.yaml", split="test", imgsz=640, batch=16, device=1)
EOF
```

看每段的 crack 行（class 1）的 AP50 / AP50-95 / Recall（P 列前两项）。

## 6. 回传结果（本地执行）

```bash
scp jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/tile_learnability_eval.log "D:/教务处实习/wood_preproject/outputs/"
scp jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/tile_learnability_yolo/conversion_summary.json "D:/教务处实习/wood_preproject/outputs/"
scp jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/tile_learnability_finetune/results.csv "D:/教务处实习/wood_preproject/outputs/tile_learnability_finetune_results.csv"
scp jzf@10.87.1.64:/home/jzf/dataset-clean/wood-multitask-final/outputs/defect_baseline/runs/tile_learnability_finetune/args.yaml "D:/教务处实习/wood_preproject/outputs/tile_learnability_finetune_args.yaml"
```

## 7. 结果解读（回传后我会做完整对比）

**核心判断**（主要看 test + ext_test + val 三个独立集合的 crack 指标是否一致地高/低）：

| 结果 | 结论 |
|---|---|
| holdout tile 上 crack AP50/Recall 明显高 | 裂纹在 640 原生尺度下**能看清**，裂纹中心 tile **可用**（可作训练数据，也可作部署时的推理单元） |
| 三个集合都很差 | 裂纹在原尺度下不可分辨（太细/对比度太低/或标注问题），与多任务架构无关，需要更大放大或改标注 |
| test（32 张）与 val/ext_test（394 张）矛盾 | 以 val/ext_test 为准（统计量更大），holdout 8 张源图本身裂纹偏少 |

**与实验一对照，定位完整因果链**：

| 实验一整图桥接 | 实验二 tile | 结论 |
|---|---|---|
| 好 | 好 | 数据本身可学，整图管线可用 → 之前是**多任务竞争**的锅 |
| 差 | 好 | 裂纹可学但整图里占比太小 → 部署改为**裂纹中心 tile 推理**，裁剪图就是正确路径 |
| 差 | 差 | **数据/标注本身**是瓶颈（裂纹太细或标不准），换架构没用 |
| 好 | 差 | 少见：tile 标注过密（每 tile 平均 8~16 框）干扰检测，或 holdout 源图特殊 → 需复查标注 |

## 注意事项

1. 不要删除或改动正在跑的 speedbatch_v3 训练与 `new history` 文件。
2. 本实验只读原图/标注/tile 目录，产物全部写入 `tile_learnability_*` 新目录。
3. test 集合（32 张 tile）偏小，是 8 张 holdout 源图裂纹数量决定的；如需扩大可改用 `--holdout 12` 重建（会少 4 张源图的训练数据），**先别改，等第一批结果**。
4. 实验一（bridge_vsb2000_finetune）如果还没跑完，两个实验的 GPU 1 使用要排队。
