# LearnRec

本仓库提供经典 CTR/推荐模型的可复现实验，包括：

- 传统模型：MF、FM、FFM
- 神经网络模型：Wide & Deep (WDL)、Deep & Cross Network (DCN)
- 序列模型：Deep Interest Network (DIN)

## 依赖

- Python 3.9+
- PyTorch、scikit-learn、pandas、numpy

```bash
pip install torch pandas scikit-learn
```

## 数据集

默认使用 DeepCTR 项目公开的 `criteo_sample.txt`，脚本会自动下载到 `data/` 目录。数据包含 13 个连续特征（I1-I13）和 26 个离散特征（C1-C26）。其中 C1 作为目标物品，C2-C6 用作 DIN 的历史行为序列示例。

## 运行示例

以 FM、DCN、DIN 为例：

```bash
# FM
python -m learnrec.train --model fm --epochs 1 --batch_size 512 --sample_num 20000

# DCN
python -m learnrec.train --model dcn --epochs 1 --batch_size 256 --sample_num 20000

# DIN
python -m learnrec.train --model din --epochs 1 --batch_size 256 --sample_num 20000 --embed_dim 16
```

命令行参数：

- `--model`: 选择 `mf|fm|ffm|wdl|dcn|din`
- `--data_dir`: 数据存放目录，默认 `data/`
- `--epochs`: 训练轮数，默认 2
- `--batch_size`: batch 大小
- `--lr`: 学习率
- `--sample_num`: 从样本文件随机采样的行数
- `--embed_dim`: embedding 维度
- `--device`: `cpu` 或 `cuda`

## 结果

训练脚本会在每个 epoch 输出 loss 和验证集 AUC，方便快速验证不同模型的效果。
