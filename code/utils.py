# 公共工具：三对角求解、扩散系数、界面调和平均、边界插值、输出重构
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy.interpolate import interp1d


def thomas(a, b, c, d):
    # 解三对角系统：a 下对角(a[0]无用), b 主对角, c 上对角(c[-1]无用), d 右端
    n = len(b)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i-1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


def D_moisture(C, D0, Ea_over, D_floor):
    # 水分扩散系数 D(C)=D0*exp(-Ea_over/C)，设下限防 C→0 刚性
    C = np.maximum(C, 1e-12)
    D = D0 * np.exp(-Ea_over / C)
    return np.maximum(D, D_floor)


def harmonic(x, y):
    # 界面调和平均，保证界面通量连续
    return 2.0 * x * y / (x + y)


def make_boundary_interp(t_nodes, values):
    # 附件1时变边界的分段线性插值器（不外推，超界报错更安全故 bounds_error=False 取端值）
    return interp1d(t_nodes, values, kind='linear',
                    bounds_error=False, fill_value=(values[0], values[-1]))


def build_grid(N, R):
    # 单元中心 FVM 网格：面在 j*dr, 单元中心在 (i+0.5)*dr
    dr = R / N
    faces = np.linspace(0.0, R, N + 1)           # f_0=0 .. f_N=R
    centers = (faces[:-1] + faces[1:]) / 2.0      # r_i
    V = (faces[1:]**2 - faces[:-1]**2) / 2.0      # 柱坐标控制体权重
    return dr, faces, centers, V


def reconstruct_center(field0, field1):
    # r=0 处偶函数二次重构：T(0)=(9*T0 - T1)/8（零梯度对称）
    return (9.0 * field0 - field1) / 8.0


def interp_to_output(centers, cell_vals, R, val0, valR, out_r_m):
    # 把单元中心解插值到输出网格（增广 r=0 与 r=R 端点后线性插值）
    aug_r = np.concatenate(([0.0], centers, [R]))
    aug_v = np.concatenate(([val0], cell_vals, [valR]))
    return np.interp(out_r_m, aug_r, aug_v)

