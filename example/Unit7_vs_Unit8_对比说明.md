# Unit7.py vs Unit8.py 对比说明

## 一、项目概述

两个文件都是基于 PyTorch 框架，使用 **糖尿病数据集 (diabetes.csv)** 训练一个多层感知机（MLP）二分类模型。Unit7 采用传统的全批量梯度下降方式，Unit8 则在其基础上引入了 **Mini-Batch** 训练机制。

---

## 二、模型结构（相同）

两者使用完全相同的网络结构：

```
输入层 (8维) → Linear(8, 6) + Sigmoid
            → Linear(6, 4) + Sigmoid
            → Linear(4, 1) + Sigmoid  → 输出 (1维，概率值)
```

- 三层全连接网络，每层后接 Sigmoid 激活函数
- 损失函数：`BCELoss`（二元交叉熵）
- 优化器：`SGD`（随机梯度下降）

---

## 三、核心差异对比

| 对比维度 | Unit7.py | Unit8.py |
|---------|----------|----------|
| **数据加载方式** | `np.loadtxt` 直接读取为 Tensor | `Dataset` 子类 + `DataLoader` 封装 |
| **训练模式** | 全批量梯度下降（Batch GD） | 小批量梯度下降（Mini-Batch SGD） |
| **每轮更新次数** | 1 次（整份数据） | N / batch_size 次（≈24 次，按 759 条数据 / 32 ≈ 24） |
| **数据打乱** | 无 | `shuffle=True`，每轮重新打乱 |
| **学习率** | `lr=0.1` | `lr=0.01` |
| **多线程加载** | 不涉及 | `num_workers=0`（可配置多线程） |
| **主程序保护** | 无 | `if __name__ == '__main__':` |
| **打印粒度** | 每 epoch 打印一次 loss | 每 epoch 内每个 batch 打印一次 loss |
| **可视化** | 有 `matplotlib` 绘制 loss 曲线 | 无 |

---

## 四、详细分析

### 4.1 数据加载方式

**Unit7.py：**
```python
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
```
- 直接将整个 CSV 文件读入内存，转为 Tensor
- 简单直接，适合小数据集
- 缺点：无法利用 PyTorch 的数据加载特性（打乱、分批次、多线程等）

**Unit8.py：**
```python
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
```
- 继承 `torch.utils.data.Dataset`，实现 `__getitem__`、`__len__` 三个必要方法
- 通过 `DataLoader` 自动完成批次划分、数据打乱、并行加载
- 是 PyTorch 推荐的标准数据加载方式，便于扩展到更大数据集

### 4.2 训练方式

**Unit7.py — 全批量梯度下降 (Batch Gradient Descent)：**
```python
for epoch in range(100):
    y_pred = model(x_data)          # 一次性计算全部数据的预测
    loss = criterion(y_pred, y_data) # 一次性计算全部数据的损失
    optimizer.zero_grad()
    loss.backward()                  # 一次性反向传播
    optimizer.step()                 # 一次性更新参数
```
- 每轮只看一次全部数据，更新一次参数
- 优点：梯度方向准确（无噪声），收敛曲线平滑
- 缺点：内存压力大，不适合大数据集；每轮只更新一次，收敛慢

**Unit8.py — 小批量随机梯度下降 (Mini-Batch SGD)：**
```python
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):  # 遍历所有 mini-batch
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
- 每轮遍历多个 batch，每个 batch 更新一次参数
- 优点：内存开销小，支持大数据集；更新频率高，收敛更快
- 引入 `shuffle=True` 打乱数据，避免模型记住样本顺序
- 缺点：梯度有噪声，loss 曲线可能有波动

### 4.3 学习率差异

| | Unit7 | Unit8 |
|---|-------|-------|
| 学习率 | 0.1 | 0.01 |

Unit8 的学习率更小，原因是：
- Mini-Batch SGD 每轮更新多次，梯度方向更"嘈杂"
- 较小的学习率可以防止单次不良梯度导致参数剧烈震荡
- 全批量梯度下降的梯度更准确，可以使用更大的学习率

### 4.4 `if __name__ == '__main__':` 保护

- Unit7 没有此保护：被 `import` 时会**自动执行**训练和绘图
- Unit8 有此保护：被 `import` 时仅定义类和函数，不会自动运行
- **Unit8 的做法更规范**，便于代码复用和模块化导入

### 4.5 多线程加载

- Unit7 不涉及
- Unit8 通过 `num_workers` 参数控制数据加载的并行线程数
  - `num_workers=0`：主进程加载（当前设置，适合 Windows 避免多进程问题）
  - `num_workers>0`：多进程并行加载，大数据集时可加速

---

## 五、总结

| | Unit7.py | Unit8.py |
|---|----------|----------|
| **适用场景** | 教学演示、小数据集快速实验 | 工程实践、大数据集训练 |
| **代码规范** | 基础 | 更规范（Dataset 封装、main 保护） |
| **可扩展性** | 差 | 好（可轻松调整 batch_size、num_workers） |
| **训练效率** | 每轮更新 1 次，收敛较慢 | 每轮更新多次，收敛更快 |
| **内存占用** | 高（一次加载全部数据） | 低（分批处理） |

**演进脉络：** Unit8 是 Unit7 的工程化升级版本，体现了从"教学代码"到"工程实践"的演进——引入 `Dataset`/`DataLoader` 机制、Mini-Batch 训练、模块保护等 PyTorch 最佳实践。
