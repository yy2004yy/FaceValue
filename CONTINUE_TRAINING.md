# 继续训练指南

## 目录结构优化

现在模型文件按模型名分类存放，结构更整洁：

```
outputs/
├── resnet18/
│   ├── checkpoints/
│   │   ├── resnet18_best.pth      # 最佳模型（推荐使用）
│   │   ├── resnet18_latest.pth    # 最新模型
│   │   └── resnet18_epoch_*.pth   # 定期保存的模型
│   └── logs/                       # TensorBoard日志
├── alexnet/
│   └── checkpoints/
└── resnext50/
    └── checkpoints/
```

## 继续训练步骤

### 方法1：从最佳模型继续训练（推荐）

如果你想从最佳模型继续训练100轮：

```bash
python train.py --resume outputs/resnet18/checkpoints/resnet18_best.pth --epochs 200
```

**说明：**
- `--resume`: 指定要恢复的checkpoint路径
- `--epochs 200`: 总共训练200轮（如果之前训练了100轮，这会继续训练到200轮）

### 方法2：从最新模型继续训练

如果你想从最新保存的模型继续：

```bash
python train.py --resume outputs/resnet18/checkpoints/resnet18_latest.pth --epochs 200
```

### 方法3：从特定epoch继续训练

如果你想从某个特定epoch继续：

```bash
python train.py --resume outputs/resnet18/checkpoints/resnet18_epoch_100.pth --epochs 200
```

## 训练参数说明

### 基本参数

```bash
python train.py \
    --model resnet18 \              # 模型类型：alexnet, resnet18, resnext50
    --resume <checkpoint_path> \    # 继续训练的checkpoint路径
    --epochs 200 \                  # 总训练轮数（会从checkpoint的epoch继续）
    --batch-size 16 \               # 批次大小
    --lr 0.001 \                    # 学习率
    --gpu 0                         # GPU设备ID（不指定则自动选择）
```

### 完整示例

```bash
# 从最佳模型继续训练，总共200轮，使用GPU 0
python train.py \
    --resume outputs/resnet18/checkpoints/resnet18_best.pth \
    --epochs 200 \
    --gpu 0
```

## 注意事项

1. **Epoch计数**：训练会从checkpoint中保存的epoch继续，所以如果checkpoint是第100轮，设置`--epochs 200`会训练到第200轮。

2. **学习率**：继续训练时会使用checkpoint中保存的优化器状态，包括当前学习率。

3. **最佳模型更新**：如果继续训练后验证集MAE更低，会自动更新`resnet18_best.pth`。

4. **日志文件**：TensorBoard日志会继续追加到同一个日志目录。

## 查看训练进度

使用TensorBoard查看训练曲线：

```bash
tensorboard --logdir outputs/resnet18/logs
```

然后在浏览器中打开 `http://localhost:6006`

## GUI使用

GUI界面已更新默认路径，会自动使用新的整洁路径：
- 默认模型路径：`outputs/resnet18/checkpoints/resnet18_best.pth`
