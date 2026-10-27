# 问题一参数（附录2常物性口径）。无 PROBLEM_FACTS.json，参数直接取自
# MODELING_REPORT.md 第十一节参数口径表 / user_data/A题.pdf 附录2，逐个命名对齐题面。
import numpy as np

# --- 物性（附录2，常数）---
RHO = 820.0            # 密度 kg/m^3
CP = 2600.0            # 比热容 J/(kg·K)
K_THERM = 0.36         # 热传导系数 W/(m·K)
ALPHA = K_THERM / (RHO * CP)   # 热扩散率 m^2/s ≈ 1.6886e-7

H_CONV = 25.0          # 对流换热系数 W/(m^2·K)
HM_MASS = 8e-7         # 对流传质系数 m/s

# --- 水分扩散系数 D(C)=D0*exp(-Ea_over/C) ---
D0 = 7e-9              # m^2/s
EA_OVER = 0.89         # 指数常数
D_FLOOR = 1e-13        # C→0 刚性护栏

# --- 几何 ---
R = 0.02               # 半径 m (2 cm)
L = 0.25               # 长度 m (25 cm)

# --- 初值 ---
T_INIT = 28.0          # ℃
C_INIT = 2.55          # kg/kg

# --- 离散/输出 ---
N_CV = 100             # 内部控制体数
DT = 1.0               # 时间步 s
T_END = 1800           # s
OUTPUT_DR_CM = 0.1     # 输出空间步 cm
OUTPUT_R_CM = np.round(np.arange(0.0, 2.0 + 1e-9, OUTPUT_DR_CM), 4)  # 0..2 cm, 21点
OUTPUT_T = np.arange(1, T_END + 1)   # 输出时间 1..1800 s（模板 A 列自1起）

# --- Picard 迭代 ---
PICARD_TOL = 1e-8
PICARD_MAXIT = 20

SEED = 0               # 确定性求解，仅规范

# --- 数据 ---
ANNEX1_PATH = 'user_data/附件1.xlsx'
ANNEX1_NROWS = 31      # t=0..1800s，每60s

# 口径一致性自检（无「参数口径表」冲突项，仅核热扩散率派生一致）
_alpha_check = K_THERM / (RHO * CP)
assert abs(_alpha_check - ALPHA) < 1e-15, "[口径冲突] 热扩散率派生不一致"
print("[params] 口径一致性 OK  alpha=%.6e" % ALPHA)
