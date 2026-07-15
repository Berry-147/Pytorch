import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
def forward(x):
    return x*w+b
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list=[]
b_list=[]
mse_list=[]

for w in np.arange(0.0,4.1,0.1):
    for b in np.arange(-2.0,2.5,0.5):
        l_sum = 0
        print('w=',w)
        print('b=',b)
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val=forward(x_val)
            loss_val=loss(x_val,y_val)
            l_sum+=loss(x_val,y_val)
            print('\t',x_val,y_pred_val,loss_val)
        print("MSE=",l_sum/3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/3)#大错特错XXXXXXXXXXXXX0分

# ================= 绘图部分修改 =================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 直接将一维列表传入 plot_trisurf
ax.plot_trisurf(w_list, b_list, mse_list, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()