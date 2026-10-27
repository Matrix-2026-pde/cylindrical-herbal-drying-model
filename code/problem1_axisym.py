# 问题一（轴向扩展）：二维轴对称 (r,z) 传热-传质求解器
# 在一维径向模型基础上加入长度方向 z：两方程各加一个轴向二阶扩散项，
# 新增 z=0 中截面对称、z=H 端面 Robin 边界（h_b=β·h, h_mb=β·h_m）。
# 圆柱+圆截面+均匀初值+周向均匀边界下，(r,z) 轴对称即三维精确形式（φ 被对称性消去）。
#
# 离散口径与一维逐项同构（见 code/problem1.py）：
#   - FVM 柱坐标，(Nr,Nz)=(100,50)，dz=H/Nz；控制体权重 V_{i}=½(f_{i+1}²-f_i²)，单元体积 V_{i}·dz。
#   - 温度：Crank–Nicolson 隐式（常物性 → 五对角稀疏矩阵时间不变，splu 一次分解）。
#   - 水分：backward-Euler + Picard（界面 D(C) 调和平均，容差 1e-8，上限 20 次），spsolve 五对角。
#   - Robin 边界用组合热阻/传质阻（单元中心↔表面↔环境）闭合，与一维完全相同。
#   - dt=1 s，0–1800 s，附件1 时变环境分段线性插值逐 1 s 驱动。
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve
from scipy.linalg import solve_banded

from params import *
import utils as u
import problem1 as p1   # 复用一维求解器与附件1读取（模块级仅设随机种子，无副作用运行）

np.random.seed(SEED)

# --- 轴向几何（params 未含半长/z 网格，此处按题面局部定义，不改 params）---
H_HALF = L / 2.0        # 半域长度 0.125 m（z=0 中截面，z=H 端面，两端条件相同）
NZ = 50                 # 轴向控制体数，dz=H/Nz=2.5 mm


def _build_connectivity(Nr, Nz):
    """预计算五对角连接结构（几何固定，与迭代无关）。idx = i*Nz + k。"""
    ii = np.arange(1, Nr)          # 径向内部界面索引 1..Nr-1
    karr = np.arange(Nz)
    IIr, KKr = np.meshgrid(ii, karr, indexing='ij')           # (Nr-1, Nz)
    rowsR_A = ((IIr - 1) * Nz + KKr).ravel()                  # 西侧单元 (i-1,k)
    rowsR_B = (IIr * Nz + KKr).ravel()                        # 东侧单元 (i,k)
    faceR_i = IIr.ravel()                                     # 界面几何索引

    iarr = np.arange(Nr)
    kk = np.arange(1, Nz)
    IIz, KKz = np.meshgrid(iarr, kk, indexing='ij')           # (Nr, Nz-1)
    rowsZ_A = (IIz * Nz + (KKz - 1)).ravel()                  # 下侧单元 (i,k-1)
    rowsZ_B = (IIz * Nz + KKz).ravel()                        # 上侧单元 (i,k)
    radZ_i = IIz.ravel()                                      # 轴向面所属径向索引

    rR_cells = (Nr - 1) * Nz + np.arange(Nz)                  # r=R 侧面单元 (i=Nr-1)
    zH_cells = np.arange(Nr) * Nz + (Nz - 1)                  # z=H 端面单元 (k=Nz-1)
    Ntot = Nr * Nz
    diag_idx = np.arange(Ntot)
    # 预拼装 COO 行列（结构固定，仅数据随迭代更新）
    rows = np.concatenate([rowsR_A, rowsR_B, rowsZ_A, rowsZ_B, diag_idx])
    cols = np.concatenate([rowsR_B, rowsR_A, rowsZ_B, rowsZ_A, diag_idx])
    return dict(Ntot=Ntot, rowsR_A=rowsR_A, rowsR_B=rowsR_B, faceR_i=faceR_i,
                rowsZ_A=rowsZ_A, rowsZ_B=rowsZ_B, radZ_i=radZ_i,
                rR_cells=rR_cells, zH_cells=zH_cells,
                coo_rows=rows, coo_cols=cols)


def _assemble(cn, gR, gZ, bcR, bcZ):
    """由面导纳装配五对角算子 M：M·φ = Σ_faces G(φ_nb-φ_cell) + 边界 -G_bc·φ_cell。"""
    Ntot = cn['Ntot']
    diag = -(np.bincount(cn['rowsR_A'], gR, Ntot) + np.bincount(cn['rowsR_B'], gR, Ntot)
             + np.bincount(cn['rowsZ_A'], gZ, Ntot) + np.bincount(cn['rowsZ_B'], gZ, Ntot)
             + np.bincount(cn['rR_cells'], bcR, Ntot) + np.bincount(cn['zH_cells'], bcZ, Ntot))
    data = np.concatenate([gR, gR, gZ, gZ, diag])
    return sp.coo_matrix((data, (cn['coo_rows'], cn['coo_cols'])), shape=(Ntot, Ntot)).tocsc()


def _solve_banded(cn, Nz, gR, gZ, bcR, bcZ, diag_add, rhs):
    """解 A·x=rhs，A=diag(diag_add)-M（M 五对角，带宽 Nz）。LAPACK 带状直接解，远快于反复稀疏 LU。

    idx=i*Nz+k：径向耦合偏移 ±Nz（值 gR），轴向耦合偏移 ±1（值 gZ）。
    """
    Ntot = cn['Ntot']
    A_diag = diag_add + (np.bincount(cn['rowsR_A'], gR, Ntot) + np.bincount(cn['rowsR_B'], gR, Ntot)
                         + np.bincount(cn['rowsZ_A'], gZ, Ntot) + np.bincount(cn['rowsZ_B'], gZ, Ntot)
                         + np.bincount(cn['rR_cells'], bcR, Ntot) + np.bincount(cn['zH_cells'], bcZ, Ntot))
    ab = np.zeros((2 * Nz + 1, Ntot))
    ab[Nz, :] = A_diag
    rA = cn['rowsR_A']                                   # 径向：rowsR_B = rA + Nz
    ab[0, rA + Nz] = -gR                                 # 上带 offset +Nz
    ab[2 * Nz, rA] = -gR                                 # 下带 offset -Nz
    zA = cn['rowsZ_A']                                   # 轴向：rowsZ_B = zA + 1
    ab[Nz - 1, zA + 1] = -gZ                             # 上带 offset +1
    ab[Nz + 1, zA] = -gZ                                 # 下带 offset -1
    return solve_banded((Nz, Nz), ab, rhs, check_finite=False, overwrite_ab=True)


def solve(Nr=N_CV, Nz=NZ, dt=DT, beta=1.0, h_conv=H_CONV, hm=HM_MASS,
          d_interface='harmonic', return_fields=True):
    """二维轴对称 (r,z) 求解 0..1800 s。beta=端面暴露系数（h_b=β·h, h_mb=β·h_m）。

    返回中截面(z=0)/端面(z=H)径向剖面时序（21点，cm）、1800 s 单元场与轴向剖面。
    """
    _, _, _, T_inf_f, Ceq_f = p1.load_boundary()
    dr, faces, centers, Vr = u.build_grid(Nr, R)   # Vr: 柱坐标径向控制体权重（每单位 z）
    dz = H_HALF / Nz
    z_centers = (np.arange(Nz) + 0.5) * dz

    Ntot = Nr * Nz
    cn = _build_connectivity(Nr, Nz)
    use_banded = Nz <= 60   # 带宽=Nz：小网格带状直接解更快，大网格改稀疏 LU（带宽无关）

    # 单元体积 V_{i,k} = Vr_i * dz（(Nr,Nz) 展平 idx=i*Nz+k）
    Vol = np.repeat(Vr * dz, Nz)

    # 径向内部界面几何权重 f_j·dz/dr（面积=2πf·dz，去掉共同 2π）；沿 z 复制
    geoR = (faces[1:Nr] * dz / dr)                  # (Nr-1,)
    geoR_full = np.repeat(geoR, Nz)                 # 展平顺序与 rowsR 对齐 (i,k)
    # 轴向内部界面几何权重 Vr_i/dz（面积权重=½(f²)差，去掉共同 2π）；沿内部 z 界面复制
    geoZ_full = np.repeat(Vr / dz, Nz - 1).reshape(Nr, Nz - 1).ravel()

    T = np.full(Ntot, T_INIT)
    C = np.full(Ntot, C_INIT)

    # ---- 温度：常物性 → M_T 时间不变，装配一次并 splu ----
    aR_T = K_THERM * geoR_full                      # 径向内部面导纳 k·f/dr
    aZ_T = K_THERM * geoZ_full                      # 轴向内部面导纳 k·Vr/dz
    U_surf = 1.0 / ((dr / 2.0) / K_THERM + 1.0 / h_conv)   # 侧面组合换热(中心↔环境)
    bcR_T = np.full(Nz, U_surf * R * dz)            # r=R 面接环境，面积权重 R·dz
    hb = beta * h_conv
    if hb > 0:
        U_end = 1.0 / ((dz / 2.0) / K_THERM + 1.0 / hb)    # 端面组合换热(中心↔环境)
        bcZ_T = U_end * Vr                          # z=H 面接环境，面积权重 Vr_i
    else:
        U_end = 0.0
        bcZ_T = np.zeros(Nr)
    M_T = _assemble(cn, aR_T, aZ_T, bcR_T, bcZ_T)
    massT = RHO * CP * Vol
    A_T = sp.diags(massT) - (dt / 2.0) * M_T
    lu_T = splu(A_T.tocsc())

    Dfun = lambda cc: u.D_moisture(cc, D0, EA_OVER, D_FLOOR)

    def src_T(Tinf):
        s = np.zeros(Ntot)
        np.add.at(s, cn['rR_cells'], bcR_T * Tinf)
        np.add.at(s, cn['zH_cells'], bcZ_T * Tinf)
        return s

    def src_C(Ceq, bcR_C, bcZ_C):
        s = np.zeros(Ntot)
        np.add.at(s, cn['rR_cells'], bcR_C * Ceq)
        np.add.at(s, cn['zH_cells'], bcZ_C * Ceq)
        return s

    nt = int(round(T_END / dt))
    out_r_m = OUTPUT_R_CM / 100.0
    nout = len(out_r_m)
    Tmid = np.empty((nt, nout)); Cmid = np.empty((nt, nout))
    Tend = np.empty((nt, nout)); Cend = np.empty((nt, nout))

    def interp_row(cell_row, val0, valR):
        return u.interp_to_output(centers, cell_row, R, val0, valR, out_r_m)

    for step in range(nt):
        tt = (step + 1) * dt
        Tinf_new = float(T_inf_f(tt)); Tinf_old = float(T_inf_f(tt - dt))
        Ceq_new = float(Ceq_f(tt))

        # ---- 温度 Crank–Nicolson: (massD - dt/2 M) T^{n+1} = massD T^n + dt/2 M T^n + dt/2 (s_old+s_new)
        rhs = massT * T + (dt / 2.0) * (M_T.dot(T) + src_T(Tinf_old) + src_T(Tinf_new))
        T = lu_T.solve(rhs)

        # ---- 水分 backward-Euler + Picard ----
        C_prev = C.copy(); C_iter = C.copy()
        for _ in range(PICARD_MAXIT):
            Dc = Dfun(C_iter)
            Dgrid = Dc.reshape(Nr, Nz)
            if d_interface == 'harmonic':
                DfaceR = u.harmonic(Dgrid[:-1, :], Dgrid[1:, :]).ravel()      # 径向界面 (Nr-1,Nz)
                DfaceZ = u.harmonic(Dgrid[:, :-1], Dgrid[:, 1:]).ravel()      # 轴向界面 (Nr,Nz-1)
            else:
                DfaceR = (0.5 * (Dgrid[:-1, :] + Dgrid[1:, :])).ravel()
                DfaceZ = (0.5 * (Dgrid[:, :-1] + Dgrid[:, 1:])).ravel()
            gR = DfaceR * geoR_full
            gZ = DfaceZ * geoZ_full
            # r=R 侧面：表面单元 D 组合传质阻
            Dsurf_R = Dgrid[Nr - 1, :]
            UmR = 1.0 / ((dr / 2.0) / Dsurf_R + 1.0 / hm)
            bcR_C = UmR * R * dz
            # z=H 端面：h_mb=β·h_m
            hmb = beta * hm
            if hmb > 0:
                Dsurf_Z = Dgrid[:, Nz - 1]
                UmZ = 1.0 / ((dz / 2.0) / Dsurf_Z + 1.0 / hmb)
                bcZ_C = UmZ * Vr
            else:
                UmZ = np.zeros(Nr); bcZ_C = np.zeros(Nr)
            rr = (Vol / dt) * C_prev + src_C(Ceq_new, bcR_C, bcZ_C)
            if use_banded:
                C_new = _solve_banded(cn, Nz, gR, gZ, bcR_C, bcZ_C, Vol / dt, rr)
            else:
                M_C = _assemble(cn, gR, gZ, bcR_C, bcZ_C)
                C_new = splu((sp.diags(Vol / dt) - M_C).tocsc()).solve(rr)
            if np.max(np.abs(C_new - C_iter)) < PICARD_TOL:
                C_iter = C_new; break
            C_iter = C_new
        C = C_iter

        # ---- 重构输出剖面（与一维同口径）----
        Tg = T.reshape(Nr, Nz); Cg = C.reshape(Nr, Nz)
        # 中截面 z=0：轴向对称，取 k=0 行（O(dz²) 即中截面值）
        row = 0
        T0 = u.reconstruct_center(Tg[0, row], Tg[1, row])
        C0 = u.reconstruct_center(Cg[0, row], Cg[1, row])
        TR = Tinf_new - (U_surf / h_conv) * (Tinf_new - Tg[Nr - 1, row])
        DsR = float(Dfun(np.array([Cg[Nr - 1, row]]))[0])
        UmR0 = 1.0 / ((dr / 2.0) / DsR + 1.0 / hm)
        CR = Ceq_new - (UmR0 / hm) * (Ceq_new - Cg[Nr - 1, row])
        Tmid[step] = interp_row(Tg[:, row], T0, TR)
        Cmid[step] = interp_row(Cg[:, row], C0, CR)
        # 端面 z=H：先 Robin 反算各径向单元的端面表面值，再径向重构
        if hb > 0:
            Tsurf_z = Tinf_new - (U_end / hb) * (Tinf_new - Tg[:, Nz - 1])
        else:
            Tsurf_z = Tg[:, Nz - 1].copy()
        if beta * hm > 0:
            DsZ = Dfun(Cg[:, Nz - 1])
            UmZ_r = 1.0 / ((dz / 2.0) / DsZ + 1.0 / (beta * hm))
            Csurf_z = Ceq_new - (UmZ_r / (beta * hm)) * (Ceq_new - Cg[:, Nz - 1])
        else:
            Csurf_z = Cg[:, Nz - 1].copy()
        Te0 = u.reconstruct_center(Tsurf_z[0], Tsurf_z[1])
        Ce0 = u.reconstruct_center(Csurf_z[0], Csurf_z[1])
        TeR = Tinf_new - (U_surf / h_conv) * (Tinf_new - Tsurf_z[Nr - 1])
        DeR = float(Dfun(np.array([Csurf_z[Nr - 1]]))[0])
        UmeR = 1.0 / ((dr / 2.0) / DeR + 1.0 / hm)
        CeR = Ceq_new - (UmeR / hm) * (Ceq_new - Csurf_z[Nr - 1])
        Tend[step] = interp_row(Tsurf_z, Te0, TeR)
        Cend[step] = interp_row(Csurf_z, Ce0, CeR)

    res = {'Tmid': Tmid, 'Cmid': Cmid, 'Tend': Tend, 'Cend': Cend,
           'z_centers_cm': (z_centers * 100.0), 'centers_cm': (centers * 100.0),
           'beta': beta, 'Nr': Nr, 'Nz': Nz}
    if return_fields:
        res['T_field'] = T.reshape(Nr, Nz).copy()
        res['C_field'] = C.reshape(Nr, Nz).copy()
    return res


# ===== 误差指标（2D vs 1D）与验证 =====
def error_series(res2d, res1d):
    """逐时刻中截面/端面径向最大偏差 E(t)（21点输出网格）。"""
    T1, C1 = res1d['T_out'], res1d['C_out']
    ET_mid = np.max(np.abs(res2d['Tmid'] - T1), axis=1)
    EC_mid = np.max(np.abs(res2d['Cmid'] - C1), axis=1)
    ET_end = np.max(np.abs(res2d['Tend'] - T1), axis=1)
    EC_end = np.max(np.abs(res2d['Cend'] - C1), axis=1)
    return ET_mid, EC_mid, ET_end, EC_end


def domain_max_and_depth(res2d, res1d):
    """1800 s 全域单元级最大偏差与轴向侵入深度（相对 1D 中心单元值，沿 z 广播）。"""
    T1c = res1d['centers_cell_T']; C1c = res1d['centers_cell_C']   # (Nr,) 单元中心解
    Tg = res2d['T_field']; Cg = res2d['C_field']
    dT = Tg - T1c[:, None]; dC = Cg - C1c[:, None]
    iT, kT = np.unravel_index(np.argmax(np.abs(dT)), dT.shape)
    iC, kC = np.unravel_index(np.argmax(np.abs(dC)), dC.shape)
    zc = res2d['z_centers_cm']
    # 侵入深度：从端面 z=H 起，|Δ|>阈值 的最深 z（换算成"距端面深度 = H - z"）
    H_cm = H_HALF * 100.0
    maxT_by_z = np.max(np.abs(dT), axis=0)   # 各 z 层全 r 最大
    maxC_by_z = np.max(np.abs(dC), axis=0)
    depthT = H_cm - zc[maxT_by_z > 0.05].min() if np.any(maxT_by_z > 0.05) else 0.0
    depthC = H_cm - zc[maxC_by_z > 0.01].min() if np.any(maxC_by_z > 0.01) else 0.0
    return {
        'dT_max': float(np.abs(dT).max()), 'dT_at': [float(res2d['centers_cm'][iT]), float(zc[kT])],
        'dC_max': float(np.abs(dC).max()), 'dC_at': [float(res2d['centers_cm'][iC]), float(zc[kC])],
        'depth_T_gt0.05_cm': float(depthT), 'depth_C_gt0.01_cm': float(depthC),
    }


def validate_constraints(res2d, T_inf_max):
    v = []
    for k in ('Tmid', 'Cmid', 'Tend', 'Cend'):
        if not np.all(np.isfinite(res2d[k])):
            v.append("FAIL %s 含 NaN/Inf" % k)
    tol = 1e-3
    Tall = np.concatenate([res2d['Tmid'].ravel(), res2d['Tend'].ravel()])
    Call = np.concatenate([res2d['Cmid'].ravel(), res2d['Cend'].ravel()])
    if Tall.min() < T_INIT - tol or Tall.max() > T_inf_max + 0.5:
        v.append("FAIL 温度越界 [%.4f,%.4f]" % (Tall.min(), Tall.max()))
    if Call.max() > C_INIT + tol or Call.min() < -tol:
        v.append("FAIL 含水率越界 [%.5f,%.5f]" % (Call.min(), Call.max()))
    # 端面偏差应 ≥ 中截面偏差（轴向影响集中端面）
    ok = len(v) == 0
    print("[validate_constraints] %s (n=%d)" % ("PASS" if ok else "FAIL", len(v)))
    for x in v[:5]:
        print("   ", x)
    return ok


def validate_capability(res_b0, res1d, res_c1):
    """能力：β=0 端面绝热必须逐位复现一维；β=1 端面必须比中截面偏差大且非零。"""
    ETm, ECm, ETe, ECe = error_series(res_b0, res1d)
    reg = max(ETe.max(), ETm.max(), ECe.max(), ECm.max())
    assert reg < 5e-3, "[能力不成立] β=0 未复现一维（max E=%.2e）" % reg
    ETm1, ECm1, ETe1, ECe1 = error_series(res_c1, res1d)
    assert ETm1.max() < 1e-2 and ECm1.max() < 1e-2, "[能力可疑] β=1 中截面偏差过大"
    assert ETe1[-1] > ETm1[-1] + 0.5, "[能力不成立] β=1 端面温度偏差未大于中截面"
    assert ECe1[-1] > 0.1, "[能力不成立] β=1 端面失水偏差过小，端面传质未真实驱动"
    print("[validate_capability] PASS  β0_reg=%.2e  β1_end_ET(1800)=%.3f  β1_end_EC(1800)=%.3f"
          % (reg, ETe1[-1], ECe1[-1]))


def _prof_1800(res2d):
    """1800 s 轴心 (r=0) 沿 z 的温度/含水率剖面（单元中心值，Nz 点）。"""
    return {'z_cm': res2d['z_centers_cm'].tolist(),
            'T_axis': res2d['T_field'][0, :].tolist(),
            'C_axis': res2d['C_field'][0, :].tolist()}


def _run_scenario(task):
    """进程池任务（Windows spawn 要求模块级函数）：求解一个 β 情景或一套收敛网格。"""
    tag, b = task
    tc = time.time()
    if b is None:   # 收敛网格，β=1
        nr, nz = (50, 25) if tag == 'conv50' else (200, 100)
        rc = solve(Nr=nr, Nz=nz, beta=1.0, return_fields=True)
    else:
        rc = solve(Nr=N_CV, Nz=NZ, beta=b, return_fields=True)
    return tag, rc, time.time() - tc


if __name__ == '__main__':
    t0 = time.time()
    print("[axisym] 载入一维基准 (N=%d) ..." % N_CV)
    _, T_inf_arr, _, _, _ = p1.load_boundary()
    T_inf_max = float(T_inf_arr.max())
    res1d = p1.solve(N=N_CV, return_full=True)
    # 一维单元中心解（用于全域偏差广播）
    res1d['centers_cell_T'] = res1d['centers_cell_T']
    print("  1D: T_c=%.4f T_s=%.4f C_c=%.4f C_s=%.4f"
          % (res1d['T_center_1800'], res1d['T_surface_1800'],
             res1d['C_center_1800'], res1d['C_surface_1800']))

    # ---- 并行求解：4 个 β 情景 + 2 套收敛网格（互不相关的独立正演 → 进程池任务级并行）----
    print("[axisym] 并行求解 β=0/0.5/1/2 + 收敛网格 (50,25)/(200,100) ...")
    from multiprocessing import Pool

    scen, conv = {}, {}
    nproc = min(6, os.cpu_count() or 4)
    with Pool(nproc) as pool:
        results = pool.map(_run_scenario, [('B', 0.0), ('C05', 0.5), ('C1', 1.0), ('C2', 2.0),
                                           ('conv50', None), ('conv200', None)])
    for tag, r, wall in results:
        if tag.startswith('conv'):
            conv[tag] = r
            print("  %s done %.1fs" % (tag, wall))
        else:
            scen[tag] = r
            ETm, ECm, ETe, ECe = error_series(r, res1d)
            b = {'B': 0.0, 'C05': 0.5, 'C1': 1.0, 'C2': 2.0}[tag]
            print("  β=%-4s done %.1fs | ET_mid=%.4f EC_mid=%.4f | ET_end(1800)=%.3f EC_end(1800)=%.3f"
                  % (b, wall, ETm.max(), ECm.max(), ETe[-1], ECe[-1]))

    ok = validate_constraints(scen['C1'], T_inf_max)
    validate_capability(scen['B'], res1d, scen['C1'])
    if not ok:
        raise ValueError("约束验证失败，必须修正后重跑")

    def maxreldiff(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))

    # 主网格 C1 vs 相邻网格（同插值到21点输出网格的 1800 s 剖面）
    c1 = scen['C1']
    conv_report = {
        'Tmid_100_50_vs_50_25': maxreldiff(c1['Tmid'][-1], conv['conv50']['Tmid'][-1]),
        'Cmid_100_50_vs_50_25': maxreldiff(c1['Cmid'][-1], conv['conv50']['Cmid'][-1]),
        'Tend_100_50_vs_50_25': maxreldiff(c1['Tend'][-1], conv['conv50']['Tend'][-1]),
        'Cend_100_50_vs_50_25': maxreldiff(c1['Cend'][-1], conv['conv50']['Cend'][-1]),
        'Tmid_200_100_vs_100_50': maxreldiff(conv['conv200']['Tmid'][-1], c1['Tmid'][-1]),
        'Cmid_200_100_vs_100_50': maxreldiff(conv['conv200']['Cmid'][-1], c1['Cmid'][-1]),
        'Tend_200_100_vs_100_50': maxreldiff(conv['conv200']['Tend'][-1], c1['Tend'][-1]),
        'Cend_200_100_vs_100_50': maxreldiff(conv['conv200']['Cend'][-1], c1['Cend'][-1]),
    }
    print("  收敛(端面C): 100/50 vs 50/25=%.3e | 200/100 vs 100/50=%.3e"
          % (conv_report['Cend_100_50_vs_50_25'], conv_report['Cend_200_100_vs_100_50']))

    # ---- 端面特征值与全域偏差（C1, β=1, 1800 s）----
    dm = domain_max_and_depth(scen['C1'], res1d)
    end_axis_T = float(scen['C1']['Tend'][-1, 0])
    end_axis_C = float(scen['C1']['Cend'][-1, 0])
    end_corner_T = float(scen['C1']['Tend'][-1, -1])
    end_corner_C = float(scen['C1']['Cend'][-1, -1])
    print("  端面轴心 T=%.4f(+%.2f) C=%.4f(%.2f) | 角点 T=%.4f C=%.4f | 全域|ΔT|max=%.3f |ΔC|max=%.3f"
          % (end_axis_T, end_axis_T - res1d['T_center_1800'],
             end_axis_C, end_axis_C - res1d['C_center_1800'],
             end_corner_T, end_corner_C, dm['dT_max'], dm['dC_max']))

    # ---- β 敏感性（端面轴心偏差, 1800 s）----
    beta_sens = {}
    for tag, b in [('C05', 0.5), ('C1', 1.0), ('C2', 2.0)]:
        beta_sens[str(b)] = {
            'end_axis_T': float(scen[tag]['Tend'][-1, 0]),
            'dT': float(scen[tag]['Tend'][-1, 0] - res1d['T_center_1800']),
            'end_axis_C': float(scen[tag]['Cend'][-1, 0]),
            'dC': float(scen[tag]['Cend'][-1, 0] - res1d['C_center_1800']),
        }

    # ---- 组织输出 JSON（全精度）----
    key_t = [100, 300, 600, 900, 1200, 1500, 1800]
    key_ti = [t - 1 for t in key_t]

    def series_dict(res2d):
        ETm, ECm, ETe, ECe = error_series(res2d, res1d)
        return {
            'ET_mid': [round(float(ETm[i]), 6) for i in key_ti],
            'EC_mid': [round(float(ECm[i]), 6) for i in key_ti],
            'ET_end': [round(float(ETe[i]), 6) for i in key_ti],
            'EC_end': [round(float(ECe[i]), 6) for i in key_ti],
        }

    out = {
        'meta': {'seed': SEED, 'Nr': N_CV, 'Nz': NZ, 'dt': DT, 'T_end': T_END,
                 'H_half_m': H_HALF, 'dz_m': H_HALF / NZ,
                 'method': 'FVM axisym (r,z) + CN(T splu once) + backward-Euler/Picard(C spsolve), 5-diag sparse',
                 'beta_scenarios': [0.0, 0.5, 1.0, 2.0], 'beta_baseline': 1.0,
                 'T_inf_max': T_inf_max, 'python': sys.version.split()[0],
                 'numpy': np.__version__},
        'output_grid': {'r_cm': OUTPUT_R_CM.tolist(), 't_key_s': key_t},
        'baseline_1d': {'T_center_1800': res1d['T_center_1800'], 'T_surface_1800': res1d['T_surface_1800'],
                        'C_center_1800': res1d['C_center_1800'], 'C_surface_1800': res1d['C_surface_1800']},
        'error_series': {tag: series_dict(scen[tag]) for tag in ('B', 'C05', 'C1', 'C2')},
        'end_face_1800_C1': {
            'axis_T': end_axis_T, 'axis_T_dev': end_axis_T - res1d['T_center_1800'],
            'axis_C': end_axis_C, 'axis_C_dev': end_axis_C - res1d['C_center_1800'],
            'corner_T': end_corner_T, 'corner_T_dev': end_corner_T - res1d['T_surface_1800'],
            'corner_C': end_corner_C, 'corner_C_dev': end_corner_C - res1d['C_surface_1800'],
        },
        'domain_max_1800_C1': dm,
        'beta_sensitivity_1800': beta_sens,
        'grid_convergence': conv_report,
        'mid_profile_1800_C1': {
            'r_cm': OUTPUT_R_CM.tolist(),
            'T': np.round(scen['C1']['Tmid'][-1], 4).tolist(),
            'C': np.round(scen['C1']['Cmid'][-1], 6).tolist(),
        },
        'end_profile_1800_C1': {
            'r_cm': OUTPUT_R_CM.tolist(),
            'T': np.round(scen['C1']['Tend'][-1], 4).tolist(),
            'C': np.round(scen['C1']['Cend'][-1], 6).tolist(),
        },
        'axial_profile_1800_C1': _prof_1800(scen['C1']),
        'field_1800_C1': {
            'r_cm': scen['C1']['centers_cm'].tolist(),
            'z_cm': scen['C1']['z_centers_cm'].tolist(),
            'T': np.round(scen['C1']['T_field'], 4).tolist(),
            'C': np.round(scen['C1']['C_field'], 6).tolist(),
        },
    }
    os.makedirs('figures', exist_ok=True)
    with open('figures/problem1_axisym_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False)
    print("[axisym] 已保存 figures/problem1_axisym_results.json  总耗时 %.1fs" % (time.time() - t0))

