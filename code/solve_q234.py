# -*- coding: utf-8 -*-
# 问题2/3/4 求解引擎（修正版）：节点型有限体积 + 自适应BDF + 事件定位 + 守恒审计
# 模型：干基含水率 U、温度 T 在物质坐标 x=r/R(t) 上的耦合扩散（附录3/4 物性）：
#   ∂U/∂t = (1/(R²x))·∂(xD∂U/∂x)，ρcp·∂T/∂t = (1/(R²x))·∂(xk∂T/∂x)
#   —— 无对流项：链式法则变换项 −Ṙx·∂U/∂r 与欧拉对流项 +Ṙx·∂U/∂r 精确对消（论文 8.3 节）
#      advect_variant=True 恢复早期推导的对流项（仅变体对照，产生 48.68h 的来源）
#      dilution_variant=True 为"体积浓度守恒解释"探针（稀释项 −2ṘU/R，见论文模型空间表）
# 离散口径：
#   - 节点 x_j=j/N 落边界（表面节点即药材表面，Robin 直接施加）；控制体体积 ∫x dx 圆柱权重
#   - 面系数：水分 D(C,T) 沿节点间线性剖面做 Q 点 Gauss 积分（均匀温度下即 Kirchhoff 割线，
#     精确处理指数型 D(C)）；热导 k 调和平均（k 变化平缓，调和平均足够——已由 q8/q16 与谐均值对照验证）
#   - 轴 r=0：通量为零（控制体积口径，无需处理 1/x 奇异）
#   - 时间：scipy BDF（带稀疏 Jacobian 模式），rtol=1e-8；事件=全场最大 U 降至 0.15（dense output 求根）；
#     达标后用无事件积分检查下一整秒/整分钟的严格"<0.15"
#   - 守恒审计：水分（体积平均 U 的变化恒等于边界流失）+ 能量（含 ρcp 随 C 变化项与域收缩项）
# 结果落盘 figures/problem_{2,3,4}_results.json（与 fill_results.py 的旧格式兼容）。
import os, sys, time, json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags, bmat
import pandas as pd
import utils as u

R0 = 0.02            # 初始半径 m
T_INIT = 28.0        # ℃
C_INIT = 2.55        # kg/kg
H_CONV = 25.0        # W/(m²K)
HM = 8e-7            # m/s
C_THRESHOLD = 0.15
ANNEX1 = 'user_data/附件1.xlsx'
ANNEX2 = 'user_data/附件2.xlsx'


# ---------- 物性（附录2/3/4，保持旧 API）----------
def props(C, T, pset):
    """返回 rho, cp, k, D。pset in {a2,a3,a4}。a3/a4 的 D 用开尔文温度。"""
    C = np.maximum(C, 1e-9)
    if pset == 'a2':
        rho = np.full_like(C, 820.0)
        cp = np.full_like(C, 2600.0)
        k = np.full_like(C, 0.36)
        D = 7e-9 * np.exp(-0.89 / C)
    elif pset == 'a3':
        rho = 650.0 + 128.0 * C
        cp = 1450.0 + 2736.0 * C / (C + 1.0)
        k = 0.21 + 0.38 * C / (C + 1.0)
        TK = T + 273.15
        D = 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / TK)
    elif pset == 'a4':
        rho = 760.0 + 90.0 * C
        cp = 1850.0 + 2150.0 * C / (C + 1.0)
        k = 0.12 + 0.20 * C / (C + 1.0)
        TK = T + 273.15
        D = 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / TK)
    else:
        raise ValueError(pset)
    return rho, cp, k, D


# ---------- 环境（附件1 + 尾延拓口径）----------
def load_env(tail='nominal'):
    """tail: 'nominal'=恒温段平台 50℃/0.05（生产口径）；'last'=附件1末值恒定；'mean'=末小时均值。"""
    df = pd.read_excel(ANNEX1, sheet_name='Sheet1')
    t = df['时间'].to_numpy(float)
    Tinf = df['温度'].to_numpy(float)
    Ceq = df['水分浓度'].to_numpy(float)
    if tail == 'nominal':
        t = np.concatenate([t, [t[-1] + 1.0, 1e9]])
        Tinf = np.concatenate([Tinf, [50.0, 50.0]])
        Ceq = np.concatenate([Ceq, [0.05, 0.05]])
    elif tail == 'mean':
        m = t >= 10800.0
        t = np.concatenate([t, [t[-1] + 1.0, 1e9]])
        Tinf = np.concatenate([Tinf, [Tinf[m].mean(), Tinf[m].mean()]])
        Ceq = np.concatenate([Ceq, [Ceq[m].mean(), Ceq[m].mean()]])
    fT = u.make_boundary_interp(t, Tinf)   # 分段线性, 超界取端值
    fC = u.make_boundary_interp(t, Ceq)
    return fT, fC, float(Tinf.max()), float(Tinf[-1]), float(Ceq[-1])


# ---------- 半径（附件2）----------
def load_radius(mode='linear'):
    """返回 (fR, fRdot, tmax)。fRdot 为中心差弦斜率插值（仅变体对照/独立验证用；
    正确模型不需要 Ṙ——物质坐标方程只依赖 R(t) 本身）。"""
    df = pd.read_excel(ANNEX2, sheet_name='Sheet1')
    t = df['时间'].to_numpy(float)
    Rcm = df['半径'].to_numpy(float)
    Rm = Rcm / 100.0
    if mode == 'linear':
        fR = u.make_boundary_interp(t, Rm)
    elif mode == 'pchip':
        fR = PchipInterpolator(t, Rm, extrapolate=False)
    else:
        raise ValueError(mode)
    Rdot = np.gradient(Rm, t)
    fRdot = u.make_boundary_interp(t, Rdot)
    return fR, fRdot, t.max()


class DryingCylinder:
    """节点型有限体积的烘干圆柱求解器（固定域或随体收缩域，附录2/3/4 物性）。"""

    def __init__(self, n=640, pset='a4', fixed_radius=False, env_tail='nominal',
                 radius_mode='linear', quadrature=8, advect_variant=False,
                 dilution_variant=False, bc_moisture='robin', bc_heat='robin',
                 h_conv=H_CONV, latent_heat=False):
        self.n = n
        self.pset = pset
        self.fixed_radius = fixed_radius
        self.env_tail = env_tail
        self.quadrature = quadrature
        self.advect_variant = advect_variant
        self.dilution_variant = dilution_variant
        self.bc_moisture = bc_moisture
        self.bc_heat = bc_heat
        self.h_conv = h_conv          # 有效对流换热系数（辐射对照时可取 25+6.5）
        self.latent_heat = latent_heat
        self.L_vap = 2.4e6            # J/kg（蒸发潜热量级对照用）
        # 节点与几何
        self.x = np.linspace(0.0, 1.0, n + 1)
        self.dx = 1.0 / n
        self.faces = (self.x[:-1] + self.x[1:]) / 2.0        # 内部面位置 x_{j+1/2}
        edges = np.concatenate(([0.0], self.faces, [1.0]))
        self.vol = np.diff(edges ** 2) / 2.0                  # 控制体体积权重 ∫x dx
        # Gauss 节点/权重（沿面线性剖面）
        g, w = leggauss(quadrature)
        self.gfrac = ((g + 1.0) / 2.0)[:, None]
        self.gw = (w / 2.0)[:, None]
        # 环境与半径
        self.fT, self.fC, self.Tinf_max, self.Tinf_last, self.Ceq_last = load_env(env_tail)
        self.fR, self.fRdot, self.radius_tmax = load_radius(radius_mode)
        self.radius_mode = radius_mode
        # 雅可比稀疏模式（两场各自的块三对角 + 全场互耦合）
        m = n + 1
        band = diags([np.ones(m - 1), np.ones(m), np.ones(m - 1)], [-1, 0, 1], format='csc')
        self.jac_sparsity = bmat([[band, band], [band, band]], format='csc')

    # ---- 环境 / 半径 ----
    def _env(self, t):
        return float(self.fT(t)), float(self.fC(t))

    def _radius(self, t):
        if self.fixed_radius:
            return R0
        return float(self.fR(t))

    def _rdot(self, t):
        """中心差弦斜率（仅 advect/dilution 变体对照用）。"""
        return float(self.fRdot(t))

    # ---- 面系数 ----
    def _face_d(self, c, temp):
        """水分面系数：D(C,T) 沿节点线性剖面的 Gauss 积分（均匀温度下=Kirchhoff 割线）。"""
        cq = c[:-1][None, :] + self.gfrac * np.diff(c)[None, :]
        tq = temp[:-1][None, :] + self.gfrac * np.diff(temp)[None, :]
        _, _, _, dq = props(cq, tq, self.pset)
        return np.sum(self.gw * dq, axis=0)

    @staticmethod
    def _face_k(k):
        return 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])       # 调和平均（k 变化平缓）

    # ---- 通量与右端 ----
    def _flux(self, u, coeff_face, radius, transfer, external, bc):
        """返回面通量数组 F[0..n+1]（单位角）：内部面 x·D̄·Δu/Δx；表面按 bc 口径。"""
        flux = np.zeros(self.n + 2)
        flux[1:-1] = self.faces * coeff_face * np.diff(u) / self.dx
        if bc == 'robin':
            flux[-1] = -radius * transfer * (u[-1] - external)
        elif bc == 'dirichlet':
            # 强汇: 表面值=环境值, 半胞传导通量 F=2D(u_external-u_n)/dx
            flux[-1] = -2.0 * coeff_face[-1] * (u[-1] - external) / self.dx
        else:
            raise ValueError(bc)
        return flux

    def rhs(self, t, y):
        temp, c = y.reshape(2, self.n + 1)
        rho, cp, k, _ = props(c, temp, self.pset)
        radius = self._radius(t)
        tinf, cinf = self._env(t)
        # 温度（Robin 换热；advect 变体另加对流项——见论文 8.3）
        kf = self._face_k(k)
        ft = self._flux(temp, kf, radius, self.h_conv, tinf, self.bc_heat)
        if self.latent_heat:
            # 蒸发潜热冷却: 表面热入流减去 L·ρ_d·hm·(U_surf-C∞)（量级对照, 非生产口径）
            rho_d = rho / (1.0 + c)
            ft[-1] -= radius * self.L_vap * rho_d[-1] * HM * (c[-1] - cinf)
        rhs_t = np.diff(ft) / (radius ** 2 * self.vol)
        if self.advect_variant:
            rdot = self._rdot(t)
            # 早期推导的迎风对流项 ξṘ/R·∂ξu（Ṙ<0 迎风=右邻; 表面节点无对流）——仅变体对照
            adv = rdot / radius * self.x[:-1] * np.diff(temp) / self.dx
            rhs_t += np.concatenate([adv, [0.0]])
        rhs_t = rhs_t / (rho * cp)
        # 水分
        df = self._face_d(c, temp)
        fc = self._flux(c, df, radius, HM, cinf, self.bc_moisture)
        rhs_c = np.diff(fc) / (radius ** 2 * self.vol)
        if self.advect_variant:
            rdot = self._rdot(t)
            adv = rdot / radius * self.x[:-1] * np.diff(c) / self.dx
            rhs_c += np.concatenate([adv, [0.0]])
        if self.dilution_variant:
            rdot = self._rdot(t)
            rhs_c += -2.0 * rdot / radius * c                 # 稀释项 −2ṘU/R（体积浓度解释探针）
        return np.concatenate([rhs_t, rhs_c])

    # ---- 时间推进与事件 ----
    def run(self, rtol=1e-8, max_step=120.0, horizon=200 * 3600.0,
            stop_event=True, audit=True, output_dt=60.0):
        m = self.n + 1
        y0 = np.concatenate([np.full(m, T_INIT), np.full(m, C_INIT)])
        atol = np.concatenate([np.full(m, rtol), np.full(m, rtol * 0.01)])
        opts = dict(method='BDF', rtol=rtol, atol=atol, jac_sparsity=self.jac_sparsity,
                    dense_output=True)
        if stop_event:
            def ev(t, y):
                return np.max(y[m:]) - C_THRESHOLD
            ev.terminal = True
            ev.direction = -1
            opts['events'] = ev
        s1 = solve_ivp(self.rhs, (0.0, min(14400.0, horizon)), y0, max_step=30.0, **opts)
        if not s1.success:
            raise RuntimeError(s1.message)
        s2 = None
        if horizon > 14400.0:
            s2 = solve_ivp(self.rhs, (14400.0, horizon), s1.y[:, -1], max_step=max_step, **opts)
            if not s2.success:
                raise RuntimeError(s2.message)
        if stop_event:
            ev_arr = s2.t_events[0] if s2 is not None else s1.t_events[0]
            if len(ev_arr) == 0:
                raise RuntimeError('未触发烘干事件: horizon 内 Cmax 未降至 %.3f' % C_THRESHOLD)
            te = float(ev_arr[0])
        else:
            te = None

        def evaluate(t):
            tt = np.atleast_1d(t).astype(float)
            out = np.empty((2 * m, len(tt)))
            first = tt <= 14400.0
            if first.any():
                out[:, first] = s1.sol(tt[first])
            if (~first).any():
                out[:, ~first] = s2.sol(tt[~first])
            return out

        # 严格"<0.15"：事件后无事件积分到下一整秒/整分钟
        strict = None
        if stop_event:
            after = solve_ivp(self.rhs, (te, te + 60.0), s2.y[:, -1], method='BDF',
                              rtol=rtol, atol=atol, jac_sparsity=self.jac_sparsity,
                              t_eval=[np.ceil(te), np.ceil(te / 60) * 60])
            strict = dict(seconds=after.t.tolist(),
                          cmax=[float(np.max(after.y[m:, i])) for i in range(after.y.shape[1])])

        # 输出时间序列（Q2 用 output_dt=1s，Q3/Q4 用 60s；末行恒为事件时刻；首行=output_dt）
        if stop_event:
            times = np.concatenate([np.arange(output_dt, te, output_dt), [te]])
        else:
            times = np.arange(output_dt, horizon + 1e-9, output_dt)
        states = evaluate(times)
        temp, c = states[:m], states[m:]

        summary = dict(n=self.n, pset=self.pset, env_tail=self.env_tail,
                       fixed_radius=self.fixed_radius, quadrature=self.quadrature,
                       advect_variant=self.advect_variant,
                       dilution_variant=self.dilution_variant,
                       rtol=rtol, max_step_s=max_step,
                       event_s=te, event_h=(te / 3600.0) if te else None,
                       event_radius_cm=(self._radius(te) * 100) if te else None,
                       end_center=float(c[0, -1]), end_surface=float(c[-1, -1]),
                       strict_check=strict)
        if audit and stop_event:
            summary.update(self._audit(times, evaluate, c))
        return summary, dict(times=times, x=self.x, temp=temp.T, c=c.T,
                             radius=np.array([self._radius(t) for t in times]))

    # ---- 守恒审计（连续口径，区间 Gauss 积分）----
    def _audit(self, times, evaluate, c):
        m = self.n + 1
        mean = 2.0 * (c.T @ self.vol)                     # 体积平均 U（时间序列）
        g, w = leggauss(5)
        cum_out = 0.0
        et = np.concatenate([[0.0], times])               # 从 t=0 起积（首区间不可漏）
        for lo, hi in zip(et[:-1], et[1:]):
            qt = (lo + hi) / 2 + (hi - lo) / 2 * g
            qc = evaluate(qt)[-1]                          # 表面 U(qt)
            qr = np.array([self._radius(t) for t in qt])
            cinf = np.array([self._env(t)[1] for t in qt])
            cum_out += (hi - lo) / 2 * np.dot(w, 2.0 * HM / qr * (qc - cinf))
        balance = float(mean[-1] - C_INIT + cum_out)
        # 能量审计: E=R²Σρcp·T·V; dE/dt = h(T∞-T_surf)R + 2ṘE/R + R²Σ(d(ρcp)/dC·∂C/∂t)·T·V
        rho0, cp0, _, _ = props(np.full(m, C_INIT), np.full(m, T_INIT), self.pset)
        E0 = R0 ** 2 * float(np.sum(rho0 * cp0 * T_INIT * self.vol))
        y_end = evaluate(times[-1:])[:, 0]
        rho_e, cp_e, _, _ = props(c[:, -1], y_end[:m], self.pset)
        E_end = self._radius(times[-1]) ** 2 * float(np.sum(rho_e * cp_e * y_end[:m] * self.vol))
        heat_int = volcorr_int = comp_int = 0.0
        dt_e = 300.0
        et = np.arange(0.0, times[-1], dt_e)
        if et[-1] < times[-1]:
            et = np.concatenate([et, [times[-1]]])
        for lo, hi in zip(et[:-1], et[1:]):
            qt = (lo + hi) / 2 + (hi - lo) / 2 * g
            yq = evaluate(qt)
            tq, cq = yq[:m], yq[m:]
            qr = np.array([self._radius(t) for t in qt])
            tinf = np.array([self._env(t)[0] for t in qt])
            rho_q, cp_q, _, _ = props(cq, tq, self.pset)
            E_q = (qr ** 2) * np.sum(rho_q * cp_q * tq * self.vol[:, None], axis=0)
            heat_q = H_CONV * (tinf - tq[-1]) * qr
            qrdot = np.array([0.0 if self.fixed_radius else self._rdot(t) for t in qt])
            volcorr_q = 2.0 * qrdot / qr * E_q
            rhs_cq = np.array([self.rhs(t, yq[:, i])[m:] for i, t in enumerate(qt)]).T
            eps = 1e-6
            rp, cpp, _, _ = props(cq + eps, tq, self.pset)
            drhocp = (rp * cpp - rho_q * cp_q) / eps
            comp_q = (qr ** 2) * np.sum(drhocp * rhs_cq * tq * self.vol[:, None], axis=0)
            heat_int += (hi - lo) / 2 * np.dot(w, heat_q)
            volcorr_int += (hi - lo) / 2 * np.dot(w, volcorr_q)
            comp_int += (hi - lo) / 2 * np.dot(w, comp_q)
        energy_residual = E_end - E0 - heat_int - volcorr_int - comp_int
        return dict(water_mean_end=float(mean[-1]),
                    boundary_outflow=float(cum_out),
                    balance_residual=balance,
                    energy_residual=float(energy_residual),
                    energy_rel=abs(energy_residual) / E0)


# ---------- 固定域求解包装（问题2/3，旧 API 兼容）----------
def solve_fixed(pset='a3', dt=1.0, t_end=None, stop_below=None,
                t_max_cap=200 * 3600.0, N=640, env_tail='nominal',
                progress=None):
    """固定域烘干。dt=输出步长；stop_below 给定时事件求根并返回 t_star_s。"""
    model = DryingCylinder(n=N, pset=pset, fixed_radius=True, env_tail=env_tail)
    horizon = t_end if t_end is not None else t_max_cap
    summ, arr = model.run(rtol=1e-8, max_step=120.0, horizon=horizon,
                          stop_event=(stop_below is not None), output_dt=dt)
    times = arr['times']; temp = arr['temp']; c = arr['c']
    # 输出到固定 0..2cm/0.1cm 网格（问题2 需要温度+水分）
    out_r_m = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4) / 100.0
    T_rows = np.array([np.interp(out_r_m / R0, model.x, temp[i]) for i in range(len(times))])
    C_rows = np.array([np.interp(out_r_m / R0, model.x, c[i]) for i in range(len(times))])
    return {
        'r_cm': (out_r_m * 100.0).round(4).tolist(),
        't_s': times.tolist(),
        'T': T_rows, 'C': C_rows,
        't_star_s': summ['event_s'],
        'Tinf_max': model.Tinf_max,
        'audit': {kk: summ.get(kk) for kk in ('balance_residual', 'water_mean_end',
                                              'boundary_outflow', 'energy_rel')},
    }


# ---------- 动边界求解包装（问题4，旧 API 兼容）----------
def solve_moving(pset='a4', dt=60.0, stop_below=C_THRESHOLD,
                 t_max_cap=200 * 3600.0, N=1280, progress=None, advect=False,
                 audit=True, env_tail='nominal', radius_mode='linear',
                 dilution=False, bc_moisture='robin'):
    """随体收缩域（物质坐标 x=r/R(t)，正确模型无对流项）。
    advect=True/dilution=True 为论文 8.3 模型空间对照变体。"""
    model = DryingCylinder(n=N, pset=pset, fixed_radius=False, env_tail=env_tail,
                           radius_mode=radius_mode, quadrature=8,
                           advect_variant=advect, dilution_variant=dilution,
                           bc_moisture=bc_moisture)
    summ, arr = model.run(rtol=1e-8, max_step=120.0, horizon=t_max_cap,
                          stop_event=(stop_below is not None), audit=audit,
                          output_dt=dt)
    times = arr['times']; c = arr['c']; radius = arr['radius']
    out_r_m = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4) / 100.0
    C_rows = np.zeros((len(times), len(out_r_m)))
    for i in range(len(times)):
        ri = radius[i]
        row = np.interp(out_r_m, model.x * ri, c[i])
        row[out_r_m > ri + 1e-12] = np.nan
        C_rows[i] = row
    out = {
        'r_cm': (out_r_m * 100.0).round(4).tolist(),
        't_s': times.tolist(), 'C': C_rows,
        'C_surf': [float(v) for v in c[:, -1]],
        'R_cm': [float(v * 100) for v in radius],
        't_star_s': summ['event_s'],
        'Tinf_max': model.Tinf_max,
    }
    if audit:
        out['audit'] = {kk: summ.get(kk) for kk in ('balance_residual', 'water_mean_end',
                                                    'boundary_outflow', 'energy_rel')}
    return out


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    key_r = [0.0, 0.5, 1.0, 1.5, 2.0]
    key_idx = [int(round(x / 0.1)) for x in key_r]

    # ===== 问题2：附录3 固定域 3h，dt=1s 输出 =====
    print("[Q2] 附录3 变物性 0..3h (节点FVM+BDF) ...", flush=True)
    t0 = time.time()
    q2 = solve_fixed(pset='a3', dt=1.0, t_end=3 * 3600.0, N=640)
    print("  Q2 完成 %.1fs, 步数=%d" % (time.time() - t0, len(q2['t_s'])), flush=True)
    q2_tbl_t = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    tbl_T = {}; tbl_C = {}
    for th in q2_tbl_t:
        i = int(np.argmin(np.abs(np.array(q2['t_s']) - th * 3600.0)))
        tbl_T["%.1f" % th] = [round(float(q2['T'][i, j]), 4) for j in key_idx]
        tbl_C["%.1f" % th] = [round(float(q2['C'][i, j]), 4) for j in key_idx]
    out2 = {
        'meta': {'pset': 'a3', 'method': 'node_FVM+BDF+Gauss_face', 'env_tail': 'nominal',
                 'Tinf_max': q2['Tinf_max']},
        'r_cm': q2['r_cm'], 't_s': q2['t_s'],
        'T_field': np.round(q2['T'], 4).tolist(), 'C_field': np.round(q2['C'], 4).tolist(),
        'table_T': tbl_T, 'table_C': tbl_C, 'key_r_cm': key_r,
    }
    with open('figures/problem_2_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out2, fh, ensure_ascii=False)
    print("  已存 figures/problem_2_results.json", flush=True)

    # ===== 问题3：附录3 固定域至 C<0.15（n=1280, rtol=1e-8）=====
    print("[Q3] 附录3 长时程至全场 C<0.15 ...", flush=True)
    t0 = time.time()
    q3 = solve_fixed(pset='a3', dt=60.0, stop_below=C_THRESHOLD, N=1280)
    tstar3 = q3['t_star_s']
    print("  Q3 完成 %.1fs, t*=%.3fh  水分收支残差=%.3e" %
          (time.time() - t0, tstar3 / 3600.0, q3['audit']['balance_residual']), flush=True)
    tbl5 = {}
    ts3 = np.array(q3['t_s'])
    for th in range(6, int(tstar3 / 3600.0) + 1, 6):
        i = int(np.argmin(np.abs(ts3 - th * 3600.0)))
        tbl5["%d" % th] = [round(float(q3['C'][i, j]), 4) for j in key_idx]
    tbl5["t_star=%.3fh" % (tstar3 / 3600.0)] = [round(float(q3['C'][-1, j]), 4) for j in key_idx]
    maxC = np.max(q3['C'], axis=1).tolist()
    out3 = {
        'meta': {'pset': 'a3', 'method': 'node_FVM+BDF+Gauss_face', 'env_tail': 'nominal',
                 't_star_s': tstar3, 't_star_h': tstar3 / 3600.0,
                 'C_threshold': C_THRESHOLD, 'Tinf_max': q3['Tinf_max']},
        'r_cm': q3['r_cm'], 't_s': q3['t_s'],
        'C_field': np.round(q3['C'], 4).tolist(),
        'maxC_t': [round(x, 6) for x in maxC],
        'table_C': tbl5, 'key_r_cm': key_r,
        'audit': q3['audit'],
    }
    with open('figures/problem_3_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out3, fh, ensure_ascii=False)
    print("  已存 figures/problem_3_results.json", flush=True)

    # ===== 问题4：附录4 + 附件2 收缩域至 C<0.15（正确模型，n=1280, rtol=1e-8）=====
    print("[Q4] 附录4 收缩域(物质坐标模型) 至全场 C<0.15 ...", flush=True)
    t0 = time.time()
    q4 = solve_moving(pset='a4', dt=60.0, stop_below=C_THRESHOLD, N=1280)
    tstar4 = q4['t_star_s']
    print("  Q4 完成 %.1fs, t*_shrink=%.3fh  水分收支残差=%.3e" %
          (time.time() - t0, tstar4 / 3600.0, q4['audit']['balance_residual']), flush=True)
    tbl6 = {}
    for th in range(6, int(tstar4 / 3600.0) + 1, 6):
        i = int(np.argmin(np.abs(np.array(q4['t_s']) - th * 3600.0)))
        row = [round(float(q4['C'][i, j]), 4) if np.isfinite(q4['C'][i, j]) else None
               for j in key_idx]
        row.append(round(float(q4['C_surf'][i]), 4))
        tbl6["%d" % th] = row
    lastrow = [round(float(q4['C'][-1, j]), 4) if np.isfinite(q4['C'][-1, j]) else None
               for j in key_idx]
    lastrow.append(round(float(q4['C_surf'][-1]), 4))
    tbl6["t_star=%.3fh" % (tstar4 / 3600.0)] = lastrow
    out4 = {
        'meta': {'pset': 'a4', 'method': 'node_FVM+BDF+Gauss_face',
                 'moving_boundary': 'landau_material_no_advect', 'env_tail': 'nominal',
                 't_star_s': tstar4, 't_star_h': tstar4 / 3600.0,
                 'C_threshold': C_THRESHOLD, 'Tinf_max': q4['Tinf_max']},
        'r_cm': q4['r_cm'], 't_s': q4['t_s'],
        'C_field': [[float(x) if np.isfinite(x) else None for x in row]
                    for row in np.round(q4['C'], 4)],
        'C_surf': [round(float(x), 4) for x in q4['C_surf']],
        'R_cm': [round(float(x), 4) for x in q4['R_cm']],
        'table_C': tbl6, 'key_r_cm': key_r,
        'audit': q4['audit'],
    }
    with open('figures/problem_4_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out4, fh, ensure_ascii=False)
    print("  已存 figures/problem_4_results.json", flush=True)

    print("[cross] Q3 t*=%.3fh  Q4 t*_shrink=%.3fh  收缩加速=%s" %
          (tstar3 / 3600.0, tstar4 / 3600.0, "是" if tstar4 <= tstar3 else "否"), flush=True)
    print("[done] Q2/Q3/Q4 全部落盘。", flush=True)

