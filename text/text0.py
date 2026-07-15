import torch

# 1. 检查 CUDA 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")

# 2. 查看显卡名称
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 3. 尝试在 GPU 上创建一个张量并计算
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"Tensor calculation on GPU successful! Result device: {z.device}")
else:
    print("CUDA is not available. Running on CPU.")