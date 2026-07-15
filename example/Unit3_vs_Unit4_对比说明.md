# Unit3.py vs Unit4.py 详细对比说明

## 概述

| | **Unit3.py** | **Unit4.py** |
|---|---|---|
| 主题 | 手动实现梯度下降（PyTorch 自动求导） | 使用 PyTorch 高阶 API 实现线性回归 |
| 学习目标 | 理解 autograd 机制和手动更新参数 | 理解 `nn.Module`、损失函数、优化器的用法 |
| 代码风格 | 过程式（手动造轮子） | 面向对象（使用 PyTorch 内置模块） |
| 模型复杂度 | `y = w·x`（1 个参数，无偏置） | `y = w·x + b`（2 个参数，含偏置） |
| 训练轮数 | 100 epoch | 1000 epoch |
| 梯度下降方式 | **随机梯度下降 (SGD)**：每一条样本更新一次 | **批量梯度下降 (BGD)**：所有样本一起计算，每 epoch 更新一次 |

---

## 一、数据准备

### Unit3.py
```python
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
```
- 使用 **Python 原生列表** 存储数据
- 前向传播时需要传入单个标量值 `x`，手动逐个计算

### Unit4.py
```python
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
```
- 使用 **PyTorch 张量**，形状为 `(3, 1)`，表示 3 条样本、每条 1 个特征
- 所有数据可以一次性传入模型，利用矩阵运算加速

> **为什么 Unit4 用矩阵形式？** 因为 `nn.Linear` 要求输入形状为 `(batch_size, input_features)`，矩阵运算也更高效。

---

## 二、模型定义

### Unit3.py — 手动定义参数
```python
w = torch.tensor([1.0])          # 手动创建权重张量
w.requires_grad = True           # 手动开启梯度追踪

def forward(x):
    return x * w                 # 手动实现线性变换

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2      # 手动实现均方误差（单样本）
```
- **自己管理参数**：`w` 是一个叶子张量，需要手动设置 `requires_grad = True`
- **自己写前向传播**：简单的 `x * w`
- **自己写损失函数**：每个样本的平方误差
- 这种方式的缺点是：参数多了以后管理起来很麻烦

### Unit4.py — 使用 `nn.Module` 封装
```python
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入1维 → 输出1维

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
```
- **`nn.Module`** 是所有神经网络的基类，必须继承
- **`nn.Linear(1, 1)`** 内部自动包含 `weight` 和 `bias` 两个张量：
  - `model.linear.weight` → 权重 `w`
  - `model.linear.bias` → 偏置 `b`
- 只需要实现 `forward()` 方法，反向传播由 PyTorch 自动完成
- 可以轻松扩展为多层网络（叠加多个 `nn.Linear` + 激活函数）

---

## 三、损失函数

### Unit3.py
```python
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
```
- 手动计算**单样本**的平方误差 `(y_pred - y)²`
- 对整个数据集需要在外层循环累加才能得到 MSE
- 没有求平均，这是**逐样本**的 loss

### Unit4.py
```python
criterion = torch.nn.MSELoss(reduction='sum')
loss = criterion(y_pred, y_data)   # 一次性传入所有样本
```
- 使用 PyTorch 内置的 **`nn.MSELoss`**
- `reduction='sum'`：计算 `Σ(y_pred - y)²`，不取平均
- 如果 `reduction='mean'`（默认），则计算 `(1/n)·Σ(y_pred - y)²`
- 可以**一次性**对所有样本计算损失

---

## 四、优化器与参数更新

### Unit3.py — 手动更新
```python
# 三步手动操作
l.backward()                                    # ① 反向传播，计算梯度
w.data = w.data - 0.01 * w.grad.data            # ② 手动更新参数
w.grad.data.zero_()                             # ③ 手动清零梯度
```
- 必须自己写 `w = w - lr * grad` 的 SGD 公式
- 必须自己**逐参数**清零梯度（否则梯度会累积）
- 参数多了容易出错

### Unit4.py — 使用 `optim.SGD`
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad()   # ① 优化器自动清零所有参数的梯度
loss.backward()          # ② 反向传播
optimizer.step()         # ③ 优化器自动更新所有参数
```
- **`model.parameters()`** 自动收集所有需要训练的参数
- **`optimizer.zero_grad()`** 一次性清零所有梯度
- **`optimizer.step()`** 自动按 SGD 公式更新所有参数
- 更换优化器（Adam、RMSprop 等）只需改一行代码

---

## 五、训练循环结构

### Unit3.py — 逐样本 SGD
```python
for epoch in range(100):               # 100 轮
    for x, y in zip(x_data, y_data):   # 逐条样本遍历
        l = loss(x, y)                 # 计算单样本 loss
        l.backward()                   # 反向传播
        w.data -= 0.01 * w.grad.data   # 立即更新
        w.grad.data.zero_()            # 立即清零
```
- 每个样本都**单独**计算 loss、反向传播、更新参数
- 这就是 **随机梯度下降 (SGD)**：每次用 1 个样本来更新
- 每 epoch 更新 **3 次**（因为有 3 条数据）
- 总共更新 **300 次**

### Unit4.py — 批量梯度下降
```python
for epoch in range(1000):              # 1000 轮
    y_pred = model(x_data)             # 一次性前向传播所有样本
    loss = criterion(y_pred, y_data)   # 一次性计算整体 loss
    optimizer.zero_grad()
    loss.backward()                    # 一次性反向传播（梯度是3个样本的平均/求和）
    optimizer.step()                   # 一次性更新参数
```
- 所有样本一起计算，一起更新
- 这实际上是 **批量梯度下降 (BGD)** / **小批量梯度下降 (Mini-batch GD)**
- 每 epoch 更新 **1 次**
- 总共更新 **1000 次**

---

## 六、关键概念演进

```
Unit3.py                           Unit4.py
───────                            ───────
手动创建参数 w         ──→   nn.Linear 自动管理 weight 和 bias
手动 requires_grad     ──→   nn.Module 自动处理
手动写 forward()       ──→   重写 forward()，框架自动构建计算图
手动写 loss 函数       ──→   nn.MSELoss 内置损失函数
手动 w = w - lr*grad   ──→   optimizer.step() 自动更新
手动 grad.zero_()      ──→   optimizer.zero_grad() 批量清零
逐样本循环更新          ──→   批量数据一次性计算
```

---

## 七、输出结果对比

| | Unit3 | Unit4 |
|---|---|---|
| 训练前预测 `x=4` | `w=1.0` → y_pred = 4.0 | — |
| 训练后预测 `x=4` | y_pred ≈ 8.0（w 趋近于 2） | y_pred ≈ 8.0 |
| 最终参数 | `w ≈ 2.0` | `w ≈ 2.0`, `b ≈ 0.0` |

两者最终都能收敛到 `y = 2x`，但：
- **Unit3** 只学 `w`，因为模型 `y = w·x` 本来就没有偏置项
- **Unit4** 同时学 `w` 和 `b`，能够验证数据确实是 `y = 2x + 0`（b 会收敛到接近 0）

---

## 八、总结

| 维度 | Unit3 | Unit4 |
|---|---|---|
| **难度** | 入门，理解底层机制 | 进阶，使用业界标准写法 |
| **代码量** | ~15 行核心代码 | ~20 行，但结构清晰可扩展 |
| **可扩展性** | 差，参数多了难管理 | 好，加层/改结构很容易 |
| **灵活性** | 高，可自定义一切 | 中，受框架 API 约束 |
| **易出错度** | 高（手动清零、手动更新） | 低（框架保证正确性） |
| **适用场景** | 学习/研究/自定义算法 | 实际项目开发 |

**学习路径**：先通过 Unit3 理解**自动求导的原理**（`backward()` + 手动更新），再过渡到 Unit4 使用**标准的 PyTorch 训练范式**（`nn.Module` + 优化器 + 损失函数），这是学习 PyTorch 的正确节奏。
