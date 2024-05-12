import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
"""
# 定义x和t的范围和步长
x = np.arange(0, 1.1, 0.05)
t = np.arange(0, 1.1, 0.05)

# 使用np.meshgrid生成网格点坐标矩阵
X, T = np.meshgrid(x, t)

# 计算y
Y = (X**2-0.5*X) * np.sin(math.pi*X + 2*math.pi*T)  # traveling wave model
# add noise
noise = np.random.normal(0, 0.001, Y.shape)  # 假设噪音的标准差为0.001
Y_noisy = Y + noise

# 打开一个文件以写入
with open('traveling_wave_model.txt', 'w') as file:
    # 遍历X, T, Y的值，格式化后写入文件
    for xi, ti, yi in zip(X.flatten(), T.flatten(), Y_noisy.flatten()):
        file.write(f'{xi}, {ti}, {yi}\n')

# 绘制3D图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, Y_noisy, cmap='viridis')

ax.set_title('行波模型三维图')
ax.set_xlabel('x轴位移x')
ax.set_ylabel('时间t')
ax.set_zlabel('y方向幅度y')
plt.show()

"""
# 二阶系统的阶梯响应的示例方程
zeta = 0.5
phi = math.pi/2
omega_n = 15

t = np.arange(0, 1.01, 0.0025)

omega_d = omega_n * np.sqrt(1 - zeta**2)  # Damped natural frequency
e_term = np.exp(-zeta * omega_n * t)  # Exponential decay term

    
y = 1 - (e_term / np.sqrt(1 - zeta**2)) * np.sin(omega_d * t + phi) 










# 打开一个文件以写入
with open('second_order_undamped.txt', 'w') as file:
    # 遍历x和y的值，格式化后写入文件
    for xi, yi in zip(t, y):
        file.write(f'{xi}, {yi}\n')

# 绘制图形
plt.figure(figsize=(10, 6))
plt.scatter(t, y, label='y(t)',s=10)
plt.title('二阶系统响应数据集')
plt.xlabel('时间t')
plt.ylabel('输出函数y(t)')
plt.grid(True)
plt.legend()
plt.show()

