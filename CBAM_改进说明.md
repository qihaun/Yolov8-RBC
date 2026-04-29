# YOLOv8s + CBAM 注意力机制改进说明

## 概述

本文档记录在 YOLOv8s 目标检测模型中集成 **CBAM（Convolutional Block Attention Module，卷积块注意力模块）** 的方案。CBAM 通过串联**通道注意力**和**空间注意力**两个子模块，从两个互补维度增强特征图的判别能力。

## 改进动机

血液细胞检测场景中存在以下挑战：
1. **红细胞高密度粘连** — 相邻红细胞边界模糊，传统卷积难以区分
2. **血小板尺寸极小** — 特征图占比极小（约 0.07%），易被背景噪声淹没
3. **多类别共存** — RBC/WBC/Platelet 三类细胞形态、尺寸、颜色差异大

CBAM 的通道注意力可自适应强化判别性特征通道，空间注意力可定位关键区域，特别适合上述场景。

## CBAM 原理

### 通道注意力（Channel Attention）
```
输入特征 F ∈ R^(C×H×W)
  → 全局平均池化 → FC → ReLU → FC → Sigmoid
  → 输出通道权重 M_c ∈ R^(C×1×1)
  → F' = M_c ⊙ F
```

### 空间注意力（Spatial Attention）
```
输入特征 F' ∈ R^(C×H×W)
  → 沿通道维度: [AvgPool(F'), MaxPool(F')]
  → Conv(7×7) → Sigmoid
  → 输出空间权重 M_s ∈ R^(1×H×W)
  → F'' = M_s ⊙ F'
```

### CBAM 整体流程
```
CBAM(x) = SpatialAttention(ChannelAttention(x))
```

## 改动位置

### 1. 新增模块：`C2f_CBAM`

**文件：** `ultralytics/nn/modules/block.py`（第 323-345 行）

```python
class C2f_CBAM(nn.Module):
    """CSP Bottleneck with 2 convolutions and CBAM attention."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_size=7):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        self.attn = CBAM(c2, kernel_size=kernel_size)  # ← 注意力模块

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))  # ← C2f 输出经 CBAM 增强
```

`C2f_CBAM` 继承 `C2f` 的 CSP 结构，在最终输出前插入 CBAM 模块，对拼接后的特征图进行通道+空间双重增强。

### 2. 模块注册

| 文件 | 改动 |
|---|---|
| `ultralytics/nn/modules/block.py` | 导入 CBAM，添加 `C2f_CBAM` 类，加入 `__all__` |
| `ultralytics/nn/modules/__init__.py` | 从 block 导入 `C2f_CBAM`，加入 `__all__` |
| `ultralytics/nn/tasks.py` | 导入 `C2f_CBAM`，添加到 `base_modules` 和 `repeat_modules` |

### 3. 模型配置文件

**文件：** `ultralytics/cfg/models/v8/yolov8s-cbam.yaml`

替换策略：将原始 `yolov8.yaml` 中**所有 C2f 模块**替换为 `C2f_CBAM`。

| 位置 | 层号 | 模块 | 说明 |
|---|---|---|---|
| Backbone | 2 | C2f_CBAM | 128ch，提取低级特征后注意力增强 |
| Backbone | 4 | C2f_CBAM | 256ch，P3 尺度特征增强 |
| Backbone | 6 | C2f_CBAM | 512ch，P4 尺度特征增强 |
| Backbone | 8 | C2f_CBAM | 1024ch，P5 尺度特征增强 |
| Neck | 12 | C2f_CBAM | 512ch，P4 融合特征增强 |
| Neck | 15 | C2f_CBAM | 256ch，P3 融合特征增强 |
| Neck | 18 | C2f_CBAM | 512ch，P4 融合特征增强 |
| Neck | 21 | C2f_CBAM | 1024ch，P5 融合特征增强 |

**共计 8 个 CBAM 模块**，覆盖骨干网络（4个）和颈部网络（4个），实现从低级到高级特征的全程注意力增强。

### 4. 参数量变化

| 模型 | 参数量 | GFLOPs | 层数 |
|---|---|---|---|---|
| YOLOv8s | 11.13M | 28.4 | 73 |
| YOLOv8s-CBAM | 11.89M | 29.1 | 113 |

参数量增加 +0.76M (+6.8%)，计算量增加 +0.7 GFLOPs (+2.5%)。

## 训练配置

| 参数 | 值 |
|---|---|
| 模型 | YOLOv8s-CBAM |
| 数据集 | BCCD (RBC/WBC/Platelet, 3类) |
| 图像尺寸 | 800×800 |
| Epochs | 200 (早停于 Epoch 83，最佳 Epoch 32) |
| Batch size | 8 |
| 优化器 | AdamW (auto) |
| 初始学习率 | 0.001429 (auto) |
| 训练时间 | 65.8 分钟 (RTX 3050 Laptop 4GB) |
| 训练/验证/测试 | 874/99/109 张 |
| 初始学习率 | 0.0006 |
| 学习率调度 | Cosine Annealing |
| Warmup | 10 epochs |
| 数据增强 | Mosaic, HSV, Horizontal Flip |
| 早停 Patience | 30 epochs |
| 设备 | GPU (CUDA) |
| 损失函数 | BCE(分类) + CIoU(回归) |

## 训练结果（测试集）

| 指标 | 基线 YOLOv8s | YOLOv8s-CBAM | 提升 |
|---|---|---|---|
| mAP@0.5 | 0.9784 | **0.9821** | **+0.37%** |
| mAP@0.5:0.95 | 0.7140 | **0.7342** | **+2.02%** |
| RBC AP@0.5 | 0.9538 | **0.9564** | +0.26% |
| WBC AP@0.5 | 0.9950 | **0.9950** | 持平 |
| Platelet AP@0.5 | 0.9865 | **0.9948** | **+0.83%** |
| Precision | 0.9309 | **0.9406** | +0.97% |
| Recall | 0.9583 | **0.9637** | +0.54% |
| 参数量 | 11.13M | 11.89M | +6.8% |
| GFLOPs | 28.4 | 29.1 | +2.5% |
| 推理速度 | 13.4ms | 14.1ms | +5.2% |

### 关键发现

1. **mAP@0.5:0.95 提升最显著（+2.02%）**：CBAM 的空间注意力增强了边界框定位精度，使更严格的 IoU 阈值下也有更好表现
2. **血小板（Platelet）AP 提升最明显（+0.83%）**：作为小目标类别，Platelet 受益于 CBAM 的空间注意力定位能力
3. **白细胞（WBC）AP 持平**：WBC 尺寸大、特征明显，注意力增强效果有限
4. **以极小代价换取全面提升**：参数量仅增 6.8%，推理速度仅慢 0.7ms

### 模型文件位置

| 模型 | 路径 |
|---|---|
| 基线 YOLOv8s | `runs/detect/train7/weights/best.pt` |
| **YOLOv8s-CBAM（本次训练）** | **`runs/detect/train_cbam/weights/best.pt`** |

## 文件清单

```
ultralytics/
├── cfg/models/v8/
│   ├── yolov8.yaml              # 原始配置
│   └── yolov8s-cbam.yaml        # CBAM 改进配置 ← 新增
├── nn/
│   ├── modules/
│   │   ├── block.py              # C2f_CBAM 类 ← 修改
│   │   ├── conv.py               # CBAM/ChannelAttention/SpatialAttention (已存在)
│   │   └── __init__.py           # 导出 C2f_CBAM ← 修改
│   └── tasks.py                  # parse_model 注册 C2f_CBAM ← 修改
└── runs/detect/train_cbam/       # 训练输出目录 ← 新增
    └── weights/best.pt           # 最优模型权重
```

## 使用方法

```python
from ultralytics import YOLO

# 加载 CBAM 改进模型
model = YOLO('ultralytics/cfg/models/v8/yolov8s-cbam.yaml')

# 训练
model.train(data='BCCD.yaml', epochs=200, imgsz=800, device=0)

# 推理
model = YOLO('runs/detect/train_cbam/weights/best.pt')
results = model.predict(source='blood_smear.jpg', conf=0.25)
```
