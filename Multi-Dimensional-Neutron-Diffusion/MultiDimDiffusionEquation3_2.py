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

平板的解析解为: $C \cdot \cos (x \cdot \pi / a)$;球的解析解为: $C / r \cdot \sin (\pi \cdot r / R)$;

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

# 1. 几何参数
# 1.1 平板
# 初始化参数
k_eff = 1  # 有效增殖系数
a = 1  # 平板的宽度
B2 = (np.pi / a) ** 2  # 系统临界时的几何曲率
l = 3  # 神经网络的深度
s = 20  # 神经网络的中间层隐藏神经单元数量
Pi = 1  # 内部权重
Pb = 100  # 边界权重
Pc = 100 # 额外配置点权重
C = 0.5  # 解析解参数


# 定义解析解
def phi_analytical(x):
    return C * np.cos(x * np.pi / a)


# 定义几何网格
geom = dde.geometry.Interval(-a / 2, a / 2)


# 定义微分方程
def pde(x, phi):
    dphi_xx = dde.grad.hessian(phi, x, i=0, j=0)
    return dphi_xx + B2 * phi


# 定义边界条件
bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 扩散方程特征向量加速收敛方法验证
observe_x = np.array([0])
observe_phi0 = dde.icbc.PointSetBC(observe_x, phi_analytical(observe_x))
# 定义数据
data = dde.data.PDE(geom, pde, [bc,observe_phi0], num_domain=898, num_boundary=2, anchors=observe_x, solution=phi_analytical, num_test=100)
# 定义神经网络
layer_size = [1] + [s] * l + [1]
activation = "tanh"
# 网络初始值权重采用高斯分布随机采样
initializer = "Glorot uniform"
net = dde.nn.PFNN(layer_size, activation, initializer)
# 定义模型
model = dde.Model(data, net)
# 定义求解器
model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, Pb, Pc])
# 训练模型
losshistory, train_state = model.train(epochs=3500)
# 保存和可视化训练结果
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# 输出在 x=0 处的值(即 C)
print("Predicted value at x=0:", model.predict(np.array([0])))
