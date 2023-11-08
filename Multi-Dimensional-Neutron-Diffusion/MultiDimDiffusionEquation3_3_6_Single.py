'''
基于PINN深度机器学习技术求解 多维中子学扩散方程
本文采用下式进行 $k_{\text {eff }}$ 的搜索:
$$
\frac{1}{D v} \frac{\partial \phi(r, t)}{\partial t}=\nabla^2 \phi(r, t)+\frac{k_{\infty} / k-1}{L^2} \phi(r, t)
$$

给出 $k$ 与 $\phi\left(r, t_0\right)$ 的任意初始值, 考察经过一定时间 $t_\tau$, 即 $t>t_\tau$ 时, $\phi(r, t)$ 对 $t$ 的偏导数是否接近 0 ,
 从而进行临界判断。可设定合适的 $\varepsilon$, 若 $\partial \phi(r, t) / \partial t>\varepsilon$, 判断为超临界状态, 需要增大 $k$ 值;
 若 $\partial \phi(r, t) / \partial t<-\varepsilon$, 为次临界状态, 需要减小 $k$ 值。当 $-\varepsilon<\partial \phi(r, t) / \partial t<\varepsilon$,
 则认为系统已达临界, 此时 $k=k_{\text {eff }}, \phi\left(r, t>t_\tau\right)$ 即为稳态时系统的 $\phi(r)$ 分布

 | 算例 | 算例 5 | 算例 6 |
| :---: | :---: | :---: |
| $k_{\infty}$ | 1.00410 | 1.00010 |
| $k_{\mathrm{eff}}$ | 1.00202 | 0.99803 |
| 初始条件 | $\cos (\pi \cdot x / a)-0.4 \cos (2 \pi \cdot x / a)-0.4$ | $0.5 \cos (2 \pi \cdot x / a)+0.5$ |
| $\sigma_{\mathrm{MSE}, \mathrm{a}}$ | $8.9108 \times 10^{-7}$ | $4.4229 \times 10^{-6}$ |
| $\sigma_{\mathrm{MSE}, \mathrm{b}}$ | $3.8236 \times 10^{-6}$ | $1.8132 \times 10^{-5}$ |
| $\sigma_{\mathrm{MSE,c}}$ | $3.6584 \times 10^{-7}$ | $8.9051 \times 10^{-6}$ |
| $\sigma_{\mathrm{MEE}, a}-N(x, t)$ 在全域 $\{D \mid-0.5 \leqslant x \leqslant 0.5,0 \leqslant t \leqslant 0.015\}$ |
'''

import deepxde as dde
import numpy as np
import torch

# 初始化参数
k_eff = 1.00  # 有效增殖系数
k_inf = 1.00010  # 无穷增殖系数
epsilon = 0.001  # 临界判断阈值
a = 1  # 平板的宽度
D = 0.211e-2  # 扩散系数
v = 2.2e3  # 中子速度
L2 = 2.1037e-4  # 系统临界时的扩散长度
B2 = (np.pi / a) ** 2  # 系统临界时的几何曲率
l = 5  # 神经网络的深度
s = 20  # 神经网络的中间层隐藏神经单元数量
Pi = 50  # 初值权重
Pb = 100  # 边界权重
Pc = 100  # 额外配置点权重
C = 0.5  # 解析解参数
k_n = 1.0
l_n = 1.0
num_terms = 10  # 考虑的项数


# 定义初值条件
def phi_0(x):
    return 0.5 * (np.cos(2 * np.pi * x[:, 0:1] / a) + 1)

# 定义解析解
# def phi_analytical(x):
#         phi_x_t = 0
#         for n in range(1, num_terms + 1):
#             integral_term = 2 / a * np.trapz(phi_0(x) * np.cos((2 * n - 1) * np.pi / a * x[:, 0:1]), x[:, 0:1])
#             exponential_term = np.cos((2 * n - 1) * np.pi / a * x[:, 0:1]) * np.exp((k_n - 1) * x[:, 1:2] / l_n)
#             phi_x_t += integral_term * exponential_term
#         return phi_x_t

# 定义几何网格
geom = dde.geometry.Interval(-a / 2, a / 2)
timedomain = dde.geometry.TimeDomain(0, 0.015)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# 定义微分方程
def pde(x, phi):
    dphi_xx = dde.grad.hessian(phi, x, i=0, j=0)
    dphi_t = dde.grad.jacobian(phi, x, i=0, j=1)
    return 1 / (D * v) * dphi_t - dphi_xx - (k_inf / k_eff - 1) / L2 * phi


# 定义边界条件
bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 定义初值条件
ic = dde.IC(geomtime, lambda x: phi_0(x), lambda _, on_initial: on_initial)
# 定义数据
data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=300, num_boundary=200, num_initial=100, solution=None,
                        num_test=10000)
# 定义神经网络
layer_size = [2] + [s] * l + [1]
activation = "tanh"
# 网络初始值权重采用高斯分布随机采样
initializer = "Glorot uniform"
net = dde.nn.PFNN(layer_size, activation, initializer)
# 定义模型
model = dde.Model(data, net)
# 定义求解器
model.compile("adam", lr=0.001, loss_weights=[1, Pb, Pi])
mean_result = 1
increment_factor = 1.0  # 初始的增量因子
current_sign = 1  # 记录当前符号，初始为正
while not epsilon > mean_result > -epsilon:
    # 训练模型
    losshistory, train_state = model.train(epochs=1000)
    # 定义临界判断函数
    observe_x = np.linspace(-a / 2, a / 2, 100)
    result = []
    for x_value in observe_x:
        t_value = 0.01
        dt = 0.001
        result.append((model.predict([x_value, t_value]) - model.predict([x_value, t_value - dt])) / dt)
    mean_result = sum(result) / len(result)
    # 判断临界状态
    if mean_result > epsilon:
        k_eff += 0.0001 * increment_factor
        if current_sign == -1:
            increment_factor /= 10.0
            current_sign = 1
    if mean_result < -epsilon:
        k_eff -= 0.0001 * increment_factor
        if current_sign == 1:
            increment_factor /= 10.0
            current_sign = -1

    print("mean_result:", mean_result)
    print("k_inf:", k_inf)
    print("k_eff:", k_eff)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

