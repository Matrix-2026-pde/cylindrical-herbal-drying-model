# 问题一：一维径向轴对称 传热(CN隐式) + 传质(backward-Euler + Picard 非线性D(C)) 联解
# 有限体积法(柱坐标控制体权重), Robin 对流边界由附件1时变环境线性插值驱动。
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from params import *
import utils as u

np.random.seed(SEED)   # 确定性求解，仅规范


def load_boundary():
    # 读附件1前31行(t=0..1800s)，构造 T_inf(t)、Ceq(t) 分段线性插值器
    _all = pd.read_excel(ANNEX1_PATH, sheet_name=None)   # 显式读全部sheet
    df = _all['Sheet1'].iloc[:ANNEX1_NROWS]
    t_nodes = df['时间'].to_numpy(float)
    T_inf = df['温度'].to_numpy(float)
    Ceq = df['水分浓度'].to_numpy(float)
    return t_nodes, T_inf, Ceq, u.make_boundary_interp(t_nodes, T_inf), u.make_boundary_interp(t_nodes, Ceq)


def solve(N=N_CV, dt=DT, h_conv=H_CONV, hm=HM_MASS, ceq_scale=1.0,
          d_interface='harmonic', bc_interp='linear', return_full=True):
    """求解 0..1800s 温度场与水分场。返回输出网格(21点,单位cm)上的时序解与关键诊断。

    离散：柱坐标FVM，界面权重 f_j=j*dr，控制体权重 V_i=(f_{i+1}^2-f_i^2)/2。
    温度：Crank–Nicolson；水分：backward-Euler + Picard 迭代更新 D(C) 界面调和平均。
    Robin 边界用组合热阻/传质阻(单元中心↔表面)闭合，等价于 -k∂T=h(T-T∞)、-D∂C=hm(C-Ceq)。
    """
    _, _, _, T_inf_f, Ceq_f = load_boundary()
    dr, faces, centers, V = u.build_grid(N, R)
    f_in = faces[1:N]              # 内部界面 f_1..f_{N-1}
    geo_in = f_in / dr             # 内部界面几何权重

    T = np.full(N, T_INIT)
    C = np.full(N, C_INIT)

    nt = int(round(T_END / dt))
    out_r_m = OUTPUT_R_CM / 100.0  # cm -> m
    T_out = np.empty((nt, len(out_r_m)))
    C_out = np.empty((nt, len(out_r_m)))

    # 温度 CN 常量系数（线性、常物性）
    aW_T = np.zeros(N); aE_T = np.zeros(N)
    aW_T[1:N] = K_THERM * faces[1:N] / dr          # west 面 (i>=1)
    aE_T[0:N-1] = K_THERM * faces[1:N] / dr        # east 面 (i<=N-2)
    U_surf = 1.0 / ((dr / 2.0) / K_THERM + 1.0 / h_conv)   # 组合换热系数(中心↔环境)
    aE_T[N-1] = U_surf * R                          # 表面单元 east 面接环境
    beta = dt / (2.0 * RHO * CP * V)               # CN 系数

    def interp_bc(tt):
        return float(T_inf_f(tt)), float(Ceq_f(tt)) * ceq_scale

    Dfun = lambda cc: u.D_moisture(cc, D0, EA_OVER, D_FLOOR)

    for step in range(nt):
        tt = (step + 1) * dt
        Tinf_new, Ceq_new = interp_bc(tt)
        Tinf_old, Ceq_old = interp_bc(tt - dt)

        # ---- 温度 Crank–Nicolson ----
        sub = np.zeros(N); diag = np.zeros(N); sup = np.zeros(N); rhs = np.zeros(N)
        # 内部+中心
        diag[:] = 1.0 + beta * (aE_T + aW_T)
        sub[1:] = -beta[1:] * aW_T[1:]
        sup[:N-1] = -beta[:N-1] * aE_T[:N-1]
        # RHS 显式部分
        rhs[:] = T.copy()
        expl = np.zeros(N)
        expl[:N-1] += aE_T[:N-1] * (np.append(T[1:], 0.0)[:N-1] - T[:N-1])
        expl[1:] += aW_T[1:] * (T[:N-1] - T[1:])
        # 表面单元 east 面接环境（覆盖上面用邻居的错误项）
        expl[N-1] = aW_T[N-1] * (T[N-2] - T[N-1]) + aE_T[N-1] * (Tinf_old - T[N-1])
        rhs[:] = T + beta * expl
        # 表面环境隐式项移到 RHS
        rhs[N-1] += beta[N-1] * aE_T[N-1] * Tinf_new
        T = u.thomas(sub, diag, sup, rhs)

        # ---- 水分 backward-Euler + Picard（非线性 D(C)）----
        gamma = dt / V
        C_prev = C.copy()
        C_iter = C.copy()
        for _ in range(PICARD_MAXIT):
            Dc = Dfun(C_iter)
            if d_interface == 'harmonic':
                Dface = u.harmonic(Dc[:N-1], Dc[1:])       # 界面调和平均
            else:
                Dface = 0.5 * (Dc[:N-1] + Dc[1:])          # 算术(敏感性备选)
            aW = np.zeros(N); aE = np.zeros(N)
            aW[1:N] = Dface * geo_in                        # 内部界面
            aE[0:N-1] = Dface * geo_in
            Dsurf = float(Dfun(np.array([C_iter[N-1]]))[0])
            Um = 1.0 / ((dr / 2.0) / Dsurf + 1.0 / hm)      # 组合传质系数
            aE[N-1] = Um * R
            s = np.zeros(N); d = np.zeros(N); p = np.zeros(N); rr = np.zeros(N)
            d[:] = 1.0 + gamma * (aE + aW)
            s[1:] = -gamma[1:] * aW[1:]
            p[:N-1] = -gamma[:N-1] * aE[:N-1]
            rr[:] = C_prev.copy()
            rr[N-1] += gamma[N-1] * aE[N-1] * Ceq_new       # 表面环境项
            C_new = u.thomas(s, d, p, rr)
            if np.max(np.abs(C_new - C_iter)) < PICARD_TOL:
                C_iter = C_new
                break
            C_iter = C_new
        C = C_iter

        # ---- 重构输出（中心偶函数二次重构，表面Robin反算）----
        T0 = u.reconstruct_center(T[0], T[1])
        C0 = u.reconstruct_center(C[0], C[1])
        TR = Tinf_new - (U_surf / h_conv) * (Tinf_new - T[N-1])
        Dsurf = float(Dfun(np.array([C[N-1]]))[0])
        Um = 1.0 / ((dr / 2.0) / Dsurf + 1.0 / hm)
        CR = Ceq_new - (Um / hm) * (Ceq_new - C[N-1])
        T_out[step] = u.interp_to_output(centers, T, R, T0, TR, out_r_m)
        C_out[step] = u.interp_to_output(centers, C, R, C0, CR, out_r_m)

    return {
        'T_out': T_out, 'C_out': C_out,
        'centers_cell_T': T.copy(), 'centers_cell_C': C.copy(),
        'T_center_1800': float(T_out[-1, 0]), 'T_surface_1800': float(T_out[-1, -1]),
        'C_center_1800': float(C_out[-1, 0]), 'C_surface_1800': float(C_out[-1, -1]),
    }


# ===== 约束验证（题设硬约束，来自 MODELING_REPORT 第十一节）=====
def validate_constraints(res, T_inf_max):
    T_out, C_out = res['T_out'], res['C_out']
    violations = []
    if not np.all(np.isfinite(T_out)) or not np.all(np.isfinite(C_out)):
        violations.append("FAIL 结果含 NaN/Inf")
    tol = 1e-4
    if T_out.min() < T_INIT - tol or T_out.max() > T_inf_max + 0.5:
        violations.append("FAIL 温度越界 [%.4f,%.4f] 应在[%.2f,%.2f]" % (T_out.min(), T_out.max(), T_INIT, T_inf_max))
    if C_out.max() > C_INIT + tol or C_out.min() < 0:
        violations.append("FAIL 含水率越界 [%.5f,%.5f] 应在[0,%.2f]" % (C_out.min(), C_out.max(), C_INIT))
    # 径向单调：T随r不减, C随r不增（逐时层, 容差1e-4）
    dT = np.diff(T_out, axis=1)
    dC = np.diff(C_out, axis=1)
    if dT.min() < -1e-3:
        violations.append("FAIL 温度径向非单调(min dT=%.2e)" % dT.min())
    if dC.max() > 1e-3:
        violations.append("FAIL 含水率径向非单调(max dC=%.2e)" % dC.max())
    # 中心温度时间单调不减
    if np.diff(T_out[:, 0]).min() < -1e-3:
        violations.append("FAIL 中心温度随时间下降")
    ok = len(violations) == 0
    print("[validate_constraints] %s  (n_violations=%d)" % ("PASS" if ok else "FAIL", len(violations)))
    for v in violations[:5]:
        print("   ", v)
    return ok


def validate_capability(res):
    # 能力：真解出时空演化的传热-传质场，非退化常数/未耦合边界
    T_out, C_out = res['T_out'], res['C_out']
    assert T_out.shape[1] == 21 and C_out.shape[1] == 21, "[能力未交付] 输出网格非21点(0..2cm每0.1)"
    assert T_out.shape[0] == 1800, "[能力未交付] 时序非1800步(每1s)"
    # 表面响应快于中心：表面升温幅度 > 中心升温幅度
    surf_rise = T_out[-1, -1] - T_INIT
    center_rise = T_out[-1, 0] - T_INIT
    assert surf_rise > center_rise + 1e-3, "[能力不成立] 表面未比中心响应更快，边界/扩散未真实作用"
    # 水分主要在表层变化：表面失水 > 中心失水
    surf_loss = C_INIT - C_out[-1, -1]
    center_loss = C_INIT - C_out[-1, 0]
    assert surf_loss > center_loss, "[能力不成立] 表层失水未大于中心，传质边界未真实驱动"
    # 中心1800s水分接近初值(τ_m≫1800s)
    assert center_loss < 0.05, "[能力可疑] 中心失水过大，1800s内中心应几乎不变(见逻辑合同bounds)"
    print("[validate_capability] PASS  surf_rise=%.3f center_rise=%.3f surf_loss=%.4f center_loss=%.4f"
          % (surf_rise, center_rise, surf_loss, center_loss))


if __name__ == '__main__':
    t0 = time.time()
    print("[problem1] 求解主网格 N=%d ..." % N_CV)
    _, T_inf_arr, _, _, _ = load_boundary()
    T_inf_max = float(T_inf_arr.max())
    res = solve(N=N_CV, return_full=True)
    print("主网格求解完成, 耗时 %.1fs" % (time.time() - t0))
    print("  T_center(1800)=%.4f  T_surface(1800)=%.4f" % (res['T_center_1800'], res['T_surface_1800']))
    print("  C_center(1800)=%.4f  C_surface(1800)=%.4f" % (res['C_center_1800'], res['C_surface_1800']))

    ok1 = validate_constraints(res, T_inf_max)
    validate_capability(res)
    if not ok1:
        raise ValueError("约束验证失败，必须修正后重跑")

    # ---- 网格加密自检 N=50/100/200 ----
    print("[problem1] 网格加密自检 N=50/100/200 ...")
    grid_conv = {}
    key_r_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
    key_idx = [int(round(x / OUTPUT_DR_CM)) for x in key_r_cm]
    for Ng in (50, 100, 200):
        rg = solve(N=Ng, return_full=True)
        grid_conv[str(Ng)] = {
            'T_1800': [round(float(rg['T_out'][-1, i]), 4) for i in key_idx],
            'C_1800': [round(float(rg['C_out'][-1, i]), 4) for i in key_idx],
        }
    # 相邻网格最大相对差
    def maxreldiff(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))
    conv_T = maxreldiff(grid_conv['200']['T_1800'], grid_conv['100']['T_1800'])
    conv_C = maxreldiff(grid_conv['200']['C_1800'], grid_conv['100']['C_1800'])
    print("  相邻网格(200 vs 100)最大相对差: T=%.3e C=%.3e" % (conv_T, conv_C))

    # ---- 逻辑方向探针：Ceq 朝更干推 δ，中心C变化应极小 ----
    res_probe = solve(N=N_CV, ceq_scale=0.98, return_full=True)
    dC_center = res_probe['C_center_1800'] - res['C_center_1800']
    probe_sign = 1 if dC_center > 1e-9 else (-1 if dC_center < -1e-9 else 0)

    # ---- 保存结果 JSON（全精度写盘，供 paper-figure）----
    out = {
        'meta': {'seed': SEED, 'N_CV': N_CV, 'dt': DT, 'T_end': T_END,
                 'method': 'FVM cylindrical + Crank-Nicolson(T) + backward-Euler/Picard(C)',
                 'T_inf_max': T_inf_max, 'python': sys.version.split()[0], 'numpy': np.__version__},
        'output_grid': {'r_cm': OUTPUT_R_CM.tolist(), 't_s': OUTPUT_T.tolist()},
        'T_field': np.round(res['T_out'], 4).tolist(),   # [1800 x 21]
        'C_field': np.round(res['C_out'], 6).tolist(),
        'key_nodes': {
            't_s': [100, 300, 600, 900, 1200, 1500, 1800],
            'r_cm': key_r_cm,
        },
        'grid_convergence': grid_conv,
        'grid_conv_maxreldiff': {'T': conv_T, 'C': conv_C},
        'logic_probes': {'bounds': [
            {'quantity': 'C_center_1800', 'claim': 'upper', 'probe_delta_sign': probe_sign,
             'delta_value': dC_center, 'note': 'Ceq朝更干推2%，中心C变化极小'}
        ]},
        'must_features': {'T_inf_t': True, 'Ceq_t': True},
    }
    # 表1/表2 关键节点数值
    tbl_t = [100, 300, 600, 900, 1200, 1500, 1800]
    tbl_idx_t = [t - 1 for t in tbl_t]   # OUTPUT_T 从1起
    out['table_T'] = {str(t): [round(float(res['T_out'][ti, i]), 4) for i in key_idx]
                      for t, ti in zip(tbl_t, tbl_idx_t)}
    out['table_C'] = {str(t): [round(float(res['C_out'][ti, i]), 4) for i in key_idx]
                      for t, ti in zip(tbl_t, tbl_idx_t)}

    os.makedirs('figures', exist_ok=True)
    with open('figures/problem_1_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False)
    print("[problem1] 已保存 figures/problem_1_results.json")
    print("  logic probe: Ceq朝干推2%%, ΔC_center=%.3e (sign=%d, 期望≈0)" % (dC_center, probe_sign))
