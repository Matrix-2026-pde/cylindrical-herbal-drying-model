# -*- coding: utf-8 -*-
# 论文图批量生成: 从 figures/*.json 与附件数据重绘正文图3~图18（TikZ 图1/2/15 在 tex 内绘制）
# 输出到 paper_figs/, 200dpi, 中文字体
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'SimHei', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.dpi': 200, 'savefig.dpi': 200,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
    'xtick.color': '#898781', 'ytick.color': '#898781',
    'axes.grid': False, 'legend.frameon': False,
})
INK, SEC, MUT, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
C_SLOT = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'magenta': '#e87ba4'}
_SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
             '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
_SEQ_ORANGE = ['#fbe3d2', '#f7c9a4', '#f2ac74', '#ed8f48', '#eb6834', '#d3521f', '#b94416',
               '#9c3808', '#7f2d05', '#632202']
cmap_T = LinearSegmentedColormap.from_list('seqT', _SEQ_BLUE)
cmap_C = LinearSegmentedColormap.from_list('seqC', _SEQ_ORANGE)

OUT = r'..\..\02_论文\figures'
os.makedirs(OUT, exist_ok=True)


def _axis(ax, xlab, ylab, xlim=None, ylim=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(SURF)
    ax.set_xlabel(xlab, color=SEC)
    ax.set_ylabel(ylab, color=SEC)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, color=GRID, lw=0.5)


def fig3_D_curve():
    C = np.linspace(0.05, 2.55, 400)
    D = 7e-9 * np.exp(-0.89 / C)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(C, D, color=C_SLOT['blue'], lw=2)
    ax.scatter([2.55, 1.51], [7e-9 * np.exp(-0.89 / 2.55), 7e-9 * np.exp(-0.89 / 1.51)],
               color=C_SLOT['orange'], zorder=3)
    ax.annotate('C=2.55: D=4.9e-09 m2/s', (2.55, 4.95e-9), xytext=(1.25, 4.6e-9),
                fontsize=7, color=SEC, arrowprops=dict(arrowstyle='-', color=MUT, lw=0.6))
    ax.annotate('C=1.51: D=3.9e-09 m2/s', (1.51, 3.9e-9), xytext=(0.55, 2.9e-9),
                fontsize=7, color=SEC, arrowprops=dict(arrowstyle='-', color=MUT, lw=0.6))
    _axis(ax, 'C / (kg·kg⁻¹)', 'D / (m²·s⁻¹)', (0, 2.55), (0, 5.2e-9))
    ax.set_title('D(C)=7×10⁻⁹ e⁻⁰·⁸⁹/ᶜ', color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_D_curve.pdf', bbox_inches='tight')
    plt.close(fig)


def fig4_env():
    df = pd.read_excel('user_data/附件1.xlsx')
    t, Tinf, Ceq = df['时间'].to_numpy(float), df['温度'].to_numpy(float), df['水分浓度'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(t / 3600.0, Tinf, color=C_SLOT['blue'], lw=2, label='T∞')
    ax2 = ax.twinx()
    ax2.plot(t / 3600.0, Ceq, color=C_SLOT['orange'], lw=2, label='Ceq')
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(colors=MUT)
    ax.set_facecolor(SURF); ax2.set_facecolor('none')
    ax.set_xlabel('t / h', color=SEC)
    ax.set_ylabel('T∞ / ℃', color=C_SLOT['blue'])
    ax2.set_ylabel('Ceq / (kg·kg⁻¹)', color=C_SLOT['orange'])
    ax.grid(True, color=GRID, lw=0.5)
    ax.axvline(0.5, color=MUT, ls='--', lw=0.8)
    ax.text(0.52, 30, '前1800 s（问题一）', fontsize=7, color=MUT)
    ax.set_xlim(0, 4)
    ax.tick_params(colors=MUT)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_env.pdf', bbox_inches='tight')
    plt.close(fig)


def fig5_6_7_q1_fields():
    d = json.load(open('figures/problem_1_results.json', encoding='utf-8'))
    T = np.array(d['T_field']); C = np.array(d['C_field'])
    r = np.array(d['output_grid']['r_cm']); t = np.arange(1, 1801)
    # 图5 温度场时空
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    im = ax.contourf(r, t / 60.0, T, levels=np.linspace(28, 37, 19), cmap=cmap_T, extend='both')
    ax.contour(r, t / 60.0, T, levels=np.linspace(28, 37, 19), colors=MUT, linewidths=0.3)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('T / ℃', color=SEC); cb.ax.tick_params(colors=MUT, labelsize=7)
    _axis(ax, 'r / cm', 't / min', (0, 2), (0, 30))
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_T_field.pdf', bbox_inches='tight')
    plt.close(fig)
    # 图6 含水率三维曲面
    fig = plt.figure(figsize=(4.6, 3.4))
    ax = fig.add_subplot(111, projection='3d')
    RR, TT = np.meshgrid(r, t / 60.0)
    ax.plot_surface(RR, TT, C, cmap=cmap_C, linewidth=0, antialiased=False, rcount=90, ccount=40)
    ax.view_init(elev=28, azim=-62)
    ax.set_xlabel('r / cm', fontsize=8, color=SEC)
    ax.set_ylabel('t / min', fontsize=8, color=SEC)
    ax.set_zlabel('C / (kg·kg⁻¹)', fontsize=8, color=SEC)
    ax.tick_params(colors=MUT, labelsize=7)
    ax.set_zlim(1.4, 2.6)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_C_surface.pdf', bbox_inches='tight')
    plt.close(fig)
    # 图7 四个时刻径向剖面
    ts = [100, 600, 1200, 1800]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.0))
    for j, tt in enumerate(ts):
        axT = axes[0, j]; axC = axes[1, j]
        axT.plot(r, T[tt - 1], color=C_SLOT['blue'], lw=2, label='T')
        axC.plot(r, C[tt - 1], color=C_SLOT['orange'], lw=2, label='C')
        axT.set_title(f'(t={tt} s)', color=INK, fontsize=8)
        for ax, q in ((axT, 'T'), (axC, 'C')):
            _axis(ax, 'r / cm', 'T / ℃' if q == 'T' else 'C / (kg·kg⁻¹)')
            ax.tick_params(labelsize=7)
            ax.set_ylim({'T': (27.9, 37.2), 'C': (1.3, 2.6)}[q])
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f'{OUT}/fig7_profiles.pdf', bbox_inches='tight')
    plt.close(fig)


def fig8_axial():
    d = json.load(open('figures/problem1_axisym_results.json', encoding='utf-8'))
    key_t = d['output_grid']['t_key_s']
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for tag, (lab, colr, ls) in [('C05', ('β=0.5', C_SLOT['aqua'], '-.')),
                                  ('C1', ('β=1', C_SLOT['magenta'], ':')),
                                  ('C2', ('β=2.0', C_SLOT['orange'], (0, (3, 1, 1, 1))))]:
        e = d['error_series'][tag]
        axes[0].plot(key_t, e['ET_mid'], color=colr, ls=ls, lw=1.5, label=lab)
        axes[1].plot(key_t, e['EC_mid'], color=colr, ls=ls, lw=1.5, label=lab)
        axes[2].plot(key_t, e['ET_end'], color=colr, ls=ls, lw=1.5, label=lab)
        axes[0].plot(key_t, e['ET_end'], color=colr, ls=ls, lw=1.5, alpha=0.25)
    _axis(axes[0], 't / s', 'E_T / ℃')
    _axis(axes[1], 't / s', 'E_C / (kg·kg⁻¹)')
    _axis(axes[2], 't / s', 'E_T_end / ℃')
    axes[0].set_title('(a) 温度偏差（实线=端面，淡线=中截面）', fontsize=8, color=INK)
    axes[1].set_title('(b) 含水率偏差', fontsize=8, color=INK)
    axes[2].set_title('(c) 端面温度偏差', fontsize=8, color=INK)
    axes[2].legend(fontsize=6.5, labelcolor=SEC)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f'{OUT}/fig8_axial_error.pdf', bbox_inches='tight')
    plt.close(fig)
    # (d) 轴向侵入深度
    zp = d['axial_profile_1800_C1']
    z = np.array(zp['z_cm']); T1 = np.array(d['mid_profile_1800_C1']['T'])
    Ta = np.array(zp['T_axis'])
    dTa = np.abs(Ta - T1[0])
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(z, dTa, color=C_SLOT['blue'], lw=2, label='|ΔT| (r=0)')
    ax.axhline(0.05, color=MUT, ls='--', lw=0.8)
    ax.annotate('4.1 cm', (12.5 - 4.1, 0.05), xytext=(8.2, 0.35), fontsize=7.5,
                color=SEC, arrowprops=dict(arrowstyle='->', color=MUT, lw=0.6))
    _axis(ax, 'z / cm', '|ΔT| / ℃', (0, 12.5), (0, 3.9))
    ax.set_title('1800 s 轴心 (r=0) 偏差沿 z 衰减 (β=1)', fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig8d_penetration.pdf', bbox_inches='tight')
    plt.close(fig)


def fig9_props():
    C = np.linspace(0.02, 2.55, 300)
    TK = 50 + 273.15
    rho3 = 650 + 128 * C; cp3 = 1450 + 2736 * C / (C + 1); k3 = 0.21 + 0.38 * C / (C + 1)
    rho4 = 760 + 90 * C; cp4 = 1850 + 2150 * C / (C + 1); k4 = 0.12 + 0.20 * C / (C + 1)
    D3 = 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / TK)
    D4 = 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / TK)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for ax, y3, y4, lab, tag in ((axes[0, 0], rho3, rho4, 'ρ (kg/m³)', '(a)'),
                                 (axes[0, 1], cp3, cp4, 'cp (J/kg/K)', '(b)'),
                                 (axes[1, 0], k3, k4, 'k (W/m/K)', '(c)')):
        ax.plot(C, y3, color=C_SLOT['blue'], lw=2, label='附录3')
        ax.plot(C, y4, color=C_SLOT['orange'], lw=2, label='附录4')
        _axis(ax, 'C / (kg·kg⁻¹)', lab)
        ax.set_title(f'{tag} {lab}', fontsize=8, color=INK)
    axes[0, 0].legend(fontsize=7, labelcolor=SEC)
    ax = axes[1, 1]
    ax.semilogy(C, D3, color=C_SLOT['blue'], lw=2, label='附录3')
    ax.semilogy(C, D4, color=C_SLOT['orange'], lw=2, label='附录4')
    _axis(ax, 'C / (kg·kg⁻¹)', 'D (m²/s, T=50℃)')
    ax.set_title('(d) D(C,T)', fontsize=8, color=INK)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(f'{OUT}/fig9_props.pdf', bbox_inches='tight')
    plt.close(fig)


def fig10_11_q2():
    d = json.load(open('figures/problem_2_results.json', encoding='utf-8'))
    T = np.array(d['T_field']); C = np.array(d['C_field'])
    t = np.array(d['t_s']) / 3600.0
    r = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].plot(t, T[:, 0], color=C_SLOT['blue'], lw=2, label='r=0')
    axes[0].plot(t, T[:, -1], color=C_SLOT['orange'], lw=2, label='r=2 cm')
    axes[1].plot(t, C[:, 0], color=C_SLOT['blue'], lw=2, label='r=0')
    axes[1].plot(t, C[:, -1], color=C_SLOT['orange'], lw=2, label='r=2 cm')
    _axis(axes[0], 't / h', 'T / ℃', (0, 3))
    _axis(axes[1], 't / h', 'C / (kg·kg⁻¹)', (0, 3))
    axes[0].set_title('(a) 温度', fontsize=9, color=INK)
    axes[1].set_title('(b) 含水率', fontsize=9, color=INK)
    axes[1].legend(fontsize=7, labelcolor=SEC)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig10_q2_center_surf.pdf', bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    im = ax.contourf(r, t, C, levels=np.linspace(1.0, 2.55, 16), cmap=cmap_C)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('C / (kg·kg⁻¹)', color=SEC); cb.ax.tick_params(colors=MUT, labelsize=7)
    _axis(ax, 'r / cm', 't / h', (0, 2), (0, 3))
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig11_q2_C_field.pdf', bbox_inches='tight')
    plt.close(fig)


def fig12_13_q3():
    d = json.load(open('figures/problem_3_results.json', encoding='utf-8'))
    t = np.array(d['t_s']) / 3600.0; r = np.array(d['r_cm'])
    C = np.array(d['C_field']); tstar = d['meta']['t_star_h']
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for j, lab in enumerate(['r=0.0 cm', 'r=0.5 cm', 'r=1.0 cm', 'r=1.5 cm', 'r=2.0 cm']):
        ax.plot(t, C[:, j], lw=2, label=lab)
    ax.axhline(0.15, color=MUT, ls='--', lw=1)
    ax.annotate('C=0.15', (1, 0.16), fontsize=8, color=SEC)
    ax.axvline(tstar, color=C_SLOT['magenta'], ls=':', lw=1)
    ax.annotate(f't*={tstar:.2f} h', (tstar + 0.5, 1.6), fontsize=8, color=SEC)
    _axis(ax, 't / h', 'C / (kg·kg⁻¹)', (0, 60), (0, 2.6))
    ax.legend(fontsize=7, labelcolor=SEC)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig12_q3_curves.pdf', bbox_inches='tight')
    plt.close(fig)
    maxC = np.array(d['maxC_t'])
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(t, maxC, color=C_SLOT['blue'], lw=2, label='Cmax(t)')
    ax.axhline(0.15, color=MUT, ls='--', lw=1)
    ax.plot([tstar], [0.15], 'o', color=C_SLOT['magenta'], ms=6, zorder=3)
    ax.annotate(f't*={tstar:.2f} h', (tstar, 0.15), xytext=(tstar * 0.55, 1.7),
                fontsize=8, color=SEC, arrowprops=dict(arrowstyle='->', color=MUT, lw=0.6))
    _axis(ax, 't / h', 'Cmax / (kg·kg⁻¹)', (0, 60), (0, 2.6))
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig13_q3_cmax.pdf', bbox_inches='tight')
    plt.close(fig)


def fig14_radius():
    df = pd.read_excel('user_data/附件2.xlsx')
    t = df['时间'].to_numpy(float) / 3600.0; R = df['半径'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(t, R, color=C_SLOT['blue'], lw=2)
    ax.axhline(2.0, color=MUT, ls='--', lw=0.8)
    ax.axhline(R[-1], color=MUT, ls='--', lw=0.8)
    ax.annotate('R0=2.00 cm', (0.5, 2.02), fontsize=8, color=SEC)
    ax.annotate(f'R={R[-1]:.3f} cm (−40.1%)', (30, R[-1] + 0.03), fontsize=8, color=SEC)
    _axis(ax, 't / h', 'R / cm', (0, 72), (1.1, 2.1))
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig14_radius.pdf', bbox_inches='tight')
    plt.close(fig)


def fig16_q4():
    d = json.load(open('figures/problem_4_results.json', encoding='utf-8'))
    t = np.array(d['t_s']) / 3600.0; r = np.array(d['r_cm'])
    C = np.array(d['C_field'], dtype=float)
    Rc = np.array(d['R_cm'])
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    CC = np.ma.masked_invalid(C)
    im = ax.contourf(r, t, CC, levels=np.linspace(0, 2.6, 14), cmap=cmap_C)
    ax.plot(Rc, t, color=INK, lw=1.4, label='r=R(t) 药材表面')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('C / (kg·kg⁻¹)', color=SEC); cb.ax.tick_params(colors=MUT, labelsize=7)
    _axis(ax, 'r / cm', 't / h', (0, 2), (0, 50))
    ax.legend(fontsize=7, labelcolor=SEC, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig16_q4_field.pdf', bbox_inches='tight')
    plt.close(fig)


def fig17_tstar():
    t3 = json.load(open('figures/problem_3_results.json', encoding='utf-8'))['meta']['t_star_h']
    t4 = json.load(open('figures/problem_4_results.json', encoding='utf-8'))['meta']['t_star_h']
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    bars = ax.bar(['问题三\n固定尺寸', '问题四\n尺寸收缩'], [t3, t4],
                  color=[C_SLOT['blue'], C_SLOT['orange']], width=0.5)
    for b, v in zip(bars, [t3, t4]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, '%.2f h' % v, ha='center', fontsize=9, color=INK)
    ax.annotate('−%.2f h (%.1f%%)' % (t3 - t4, (t3 - t4) / t3 * 100), (1.1, t3 - 3.5),
                fontsize=8.5, color=SEC, arrowprops=dict(arrowstyle='->', color=MUT, lw=0.7))
    _axis(ax, '', 't* / h', ylim=(0, t3 + 5))
    ax.tick_params(colors=MUT)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig17_tstar.pdf', bbox_inches='tight')
    plt.close(fig)


def fig18_sensitivity():
    d = json.load(open('figures/sensitivity_results.json', encoding='utf-8'))
    base = d['baseline']['C_surface_1800']
    rows = []
    p = d['parameters']
    for name, lab in (('h_conv', 'h (±10%)'), ('hm_mass', 'hm (±10%)'), ('ceq_scale', 'Ceq (±20%)')):
        vs = p[name]['C_surface_1800']
        rows.append((lab, [(vs[0] - base) / base * 100, (vs[2] - base) / base * 100]))
    rows.append(('h (±10%)\n对T_surf', None))
    Ts = d['parameters']['h_conv']['T_surface_1800']
    Tb = d['baseline']['T_surface_1800']
    rows[-1] = ('h (±10%)\n(响应T)', [(Ts[0] - Tb) / Tb * 100, (Ts[2] - Tb) / Tb * 100])
    labels = [r[0] for r in rows]
    lo = [r[1][0] for r in rows]; hi = [r[1][1] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(rows))
    ax.barh(x, lo, color=C_SLOT['blue'], height=0.5)
    ax.barh(x, hi, color=C_SLOT['orange'], height=0.5)
    ax.axvline(0, color=INK, lw=0.8)
    for i, (a, b) in enumerate(zip(lo, hi)):
        ax.text(a - 0.15, i, f'{a:+.2f}%', va='center', ha='right', fontsize=7.5, color=SEC)
        ax.text(b + 0.15, i, f'{b:+.2f}%', va='center', ha='left', fontsize=7.5, color=SEC)
    ax.set_yticks(x); ax.set_yticklabels(labels, fontsize=8, color=SEC)
    ax.set_xlabel('表面响应相对变化 / %', color=SEC)
    ax.set_xlim(-7.5, 7.5)
    ax.tick_params(colors=MUT)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.5, axis='x')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig18_sensitivity.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig3_D_curve()
    fig4_env()
    fig5_6_7_q1_fields()
    fig8_axial()
    fig9_props()
    if os.path.exists('figures/problem_2_results.json'):
        fig10_11_q2()
    fig14_radius()
    if os.path.exists('figures/problem_3_results.json'):
        fig12_13_q3()
    if os.path.exists('figures/problem_4_results.json'):
        fig16_q4()
    fig17_tstar()
    if os.path.exists('figures/sensitivity_results.json'):
        fig18_sensitivity()
    print('[make_paper_figs] done ->', OUT)

