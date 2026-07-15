import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_1 = torch.tensor([1.0])  # w的初值为1.0
w_2 = torch.tensor([1.0])  # w的初值为1.0
w_1.requires_grad = True  # 需要计算梯度
w_2.requires_grad = True  # 需要计算梯度
b = torch.tensor([0.0])
b.requires_grad = True

def forward(x):
    return x * x * w_1+x * w_2 + b  # w是一个Tensor

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        print(f'Epoch {epoch + 1}: w1 = {w_1.item()}, w2 = {w_2.item()}, b ={b.item()}, loss = {loss_val.item()}')
        w_1.data = w_1.data - 0.01 * w_1.grad.data  # 权重更新时，注意grad也是一个tensor
        w_1.grad.data.zero_()  # after update, remember set the grad to zero
        w_2.data = w_2.data - 0.01 * w_2.grad.data  # 权重更新时，注意grad也是一个tensor
        w_2.grad.data.zero_()  # after update, remember set the grad to zero
        b.data = b.data - 0.01 * b.grad.data  # 权重更新时，注意grad也是一个tensor
        b.grad.data.zero_()  # after update, remember set the grad to zero

print("predict (after training)", 4, forward(4).item())