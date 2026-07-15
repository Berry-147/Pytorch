import torch
import torch.nn as nn
import torch.optim as optim
import time


def test_gpu_environment():
    print("=" * 40)
    print("1. 基础环境检查")
    print("=" * 40)
    print(f"PyTorch 版本: {torch.__version__}")

    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！请检查显卡驱动或 PyTorch 安装版本。")
        return

    print(f"✅ CUDA 可用！")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")

    print("\n" + "=" * 40)
    print("2. 完整训练循环测试 (前向+反向传播)")
    print("=" * 40)
    device = torch.device("cuda")

    # 构建一个简单的神经网络
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    ).to(device)

    # 准备随机数据并移至 GPU
    input_data = torch.randn(64, 100).to(device)
    target = torch.randn(64, 1).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 运行 5 个 Epoch 的训练循环
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    print("✅ GPU 训练循环测试通过！")

    print("\n" + "=" * 40)
    print("3. 大规模矩阵乘法性能对比")
    print("=" * 40)
    # 创建两个大矩阵
    size = 10000
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    # CPU 计算耗时
    start_time = time.time()
    torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time

    # 将数据移至 GPU
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    # 预热 GPU（避免首次运行计入缓存和内核加载时间）
    torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # 同步等待，确保GPU计算完成

    # GPU 计算耗时
    start_time = time.time()
    torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time

    print(f"矩阵大小: {size}x{size}")
    print(f"CPU 耗时: {cpu_time:.4f} 秒")
    print(f"GPU 耗时: {gpu_time:.4f} 秒")
    print(f"🚀 加速比: {cpu_time / gpu_time:.2f} 倍")


if __name__ == "__main__":
    test_gpu_environment()