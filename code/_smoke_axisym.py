# -*- coding: utf-8 -*-
# 冒烟测试: (1) beta=0 端面绝热时 2D 中截面必须精确复现 1D; (2) beta=1 无 NaN/越界
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import problem1_axisym as ax
import problem1 as p1
from params import *

# (1) 回归: 小网格 Nr=20, Nz=10, beta=0（端面绝热应逐位等于一维）
res2d = ax.solve(Nr=20, Nz=10, beta=0.0)
res1d = p1.solve(N=20, return_full=True)
dT = np.max(np.abs(res2d['Tmid'] - res1d['T_out']))
dC = np.max(np.abs(res2d['Cmid'] - res1d['C_out']))
print("[smoke] beta=0 中截面 vs 一维: max|dT|=%.3e  max|dC|=%.3e  (期望 ~1e-12)"
      % (dT, dC))
assert dT < 1e-8 and dC < 1e-10, "回归失败: 2D(beta=0) 未复现 1D"

# (2) beta=1 小网格, 检查有限性与物理界
res2 = ax.solve(Nr=20, Nz=10, beta=1.0)
Tf, Cf = res2['T_field'], res2['C_field']
print("[smoke] beta=1 1800s: T in [%.3f, %.3f]  C in [%.4f, %.4f]  有限=%s"
      % (Tf.min(), Tf.max(), Cf.min(), Cf.max(), np.isfinite(Tf).all() and np.isfinite(Cf).all()))
assert np.isfinite(Tf).all() and np.isfinite(Cf).all()
assert Tf.min() >= T_INIT - 1e-9 and Tf.max() <= 45.0
assert Cf.max() <= C_INIT + 1e-9 and Cf.min() >= 0.0
print("[smoke] PASS")

