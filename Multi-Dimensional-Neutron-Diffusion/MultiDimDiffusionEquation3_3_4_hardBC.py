'''
基于PINN深度机器学习技术求解 多维中子学扩散方程
临界条件下稳态扩散方程的验证
当系统处于稳态时, 形式为:
$$
\nabla^2 \phi(r)+\frac{k_{\infty} / k_{\text {eff }}-1}{L^2} \phi(r)=0
$$

当系统处于临界状态 $\left(k_{\mathrm{eff}}=1\right)$ 时, 为:
$$
\nabla^2 \phi(r)+B_g^2 \phi(r)=0
$$

式中, $B_g^2$ 为系统临界时的几何曲率, 与系统的几何特性相关，临界时等于材料曲率。

为了验证计算结果, 选取针对特定几何有解析解的扩散方程进行数值验证, 相关结论也可供其他形式的方程与几何形式参考。
验证计算神经网络架构均采用全连接方式, 激活函数选取具有高阶导数连续特点的双曲正切函数 $\mathrm{tanh}$,
其形式为 $\tanh (x)=\left(\mathrm{e}^x-\mathrm{e}^{-x}\right) /\left(\mathrm{e}^x+\mathrm{e}^{-x}\right)$,
网络初始值权重 $\{\vec{w}, \vec{b}\}$采用高斯分布随机采样 。
瞬态方程式 (2) 在平板几何真空边界条件下解析解为 ${ }^{[1]}$ :
$$
\begin{aligned}
\phi(x, t)= & \sum_{n=1}^{\infty} \frac{2}{a}\left[\int_{-a / 2}^{a / 2} \phi_0\left(x^{\prime}\right) \cos \frac{(2 n-1) \pi}{a} x^{\prime} \mathrm{d} x^{\prime}\right] \times \\
& \cos \frac{(2 n-1) \pi}{a} x \mathrm{e}^{\left(k_n-1\right) t / l_n}
\end{aligned}
$$

验证模型方程参数: $v=2.2 \times 10^3 \mathrm{~m} / \mathrm{s}, D=0.211 \times$ $10^{-2} \mathrm{~m}, L^2=2.1037 \times 10^{-4} \mathrm{~m}^2, a=1 \mathrm{~m}$,
几何网格点数共 6000 , 其中 3000 点均匀分布, 其余 3000 网格点在边界条件区域、初始条件区域进行局部加密, 其他超参数选择与 3.1 节一致。不同的 $k_{\infty}$值与不同初始条件下, 训练 1500 次后,
验证计算结果如表 3 所示。4 个算例在定义区域内 $\{D \mid$ $-0.5 \leqslant x \leqslant 0.5,0 \leqslant t \leqslant 0.015\}$ 的神经网络模型泛化计算结果 $N(\vec{x})$ 的散点图

| 算例 | $k_{\infty}$ | 初始条件 | $\sigma_{\mathrm{MSE}, 1}$ |
| :---: | :---: | :---: | :---: |
| 1 | 1.0041 | $\cos (\pi \cdot x / a)-0.4 \cos (2 \pi \cdot x / a)-0.4$ | $3.8005 \times 10^{-7}$ |
| 2 | 1.0001 | $\cos (\pi \cdot x / a)-0.4 \cos (2 \pi \cdot x / a)-0.4$ | $6.3758 \times 10^{-7}$ |
| 3 | 1.0041 | $0.5[\cos (2 \pi \cdot x / a)+1]$ | $1.5564 \times 10^{-6}$ |
| 4 | 1.0001 | $0.5[\cos (2 \pi \cdot x / a)+1]$ | $1.2230 \times 10^{-6}$ |

验证计算神经网络的超参数设定为: 深度 $l=16$, 中间层隐藏神经单元数量 $s=20$, 边界权重 $P_{\mathrm{b}}=100, C=0.5$,
几何网格点随机均布, 学习率从 0.001 开始, 训练至损失函数值 $f_{\text {Loss }}$ 在 100 次学习内不再下降结束.

3.2 扩散方程特征向量加速收敛方法验证

验证计算的目标是：统计并分析不同初始权重值 $\{\vec{w}, \vec{b}\}$ 的神经网络 $N(\vec{x})$, 在训练到相似精度时,所需要的收敛时间。

算例 1、算例 2 网络初始值权重 $\{\vec{w}, \vec{b}\}$ 随机选择 ${ }^{[12]}$; 算例 3 也选择随机初始值,
但将 $x=0$ 时, $\phi(0)$ 值 [ 式 (10) 解析解中的 $C$ 值 ] 设定为 0.5 ,如 2.1 节所述作为 $f_{\text {Loss }}$ 的组成部分;
算例 4 、算例 5、算例 6 初始状态为具有不同 $C_0$ 值、已经事先训练好的精度小于 $10^{-7}$ 网络, 训练方式与算例 3 类似,
将 $\phi(0)=0.5$ 作为加权损失函数组成部分进行训练。各算例训练精度小于 $10^{-7}$ 即停止, 记录所需要的训练次数及精度等相关参数, 结果如表 2 所示。
'''

import deepxde as dde
import numpy as np

# 初始化参数
k_eff = 1  # 有效增殖系数
k_inf = 1.0001  # 无穷增殖系数
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
    return 1 / (D * v) * dphi_t - dphi_xx - (k_inf - 1) / L2 * phi


# 定义边界条件
# bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 定义初值条件
ic = dde.IC(geomtime, lambda x: phi_0(x), lambda _, on_initial: on_initial)
# 定义数据
data = dde.data.TimePDE(geomtime, pde, [ic], num_domain=300, num_boundary=200, num_initial=100, solution=None,
                        num_test=10000)
# 定义神经网络
layer_size = [2] + [s] * l + [1]
activation = "tanh"
# 网络初始值权重采用高斯分布随机采样
initializer = "Glorot uniform"
net = dde.nn.PFNN(layer_size, activation, initializer)


# 定义硬边界条件
def output_transform(x, y):
    return (x[:, 0:1] + a / 2) * (x[:, 0:1] - a / 2) * y


net.apply_output_transform(output_transform)
# 定义模型
model = dde.Model(data, net)
# 定义求解器
model.compile("adam", lr=0.001, loss_weights=[1, Pi])
# 训练模型
losshistory, train_state = model.train(epochs=5000)
# 保存和可视化训练结果
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
