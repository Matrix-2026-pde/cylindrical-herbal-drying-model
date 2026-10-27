# -*- coding: utf-8 -*-
# 轴向影响验证: 汇总各算例 JSON -> 汇总表 JSON + 4 张论文图
# 色板: dataviz 参考色板(light), 类别固定顺序 蓝/橙/青/品红; 顺序单色相; 发散蓝-灰-红
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'SimHei', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.dpi': 200, 'savefig.dpi': 200,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
    'xtick.color': '#898781', 'ytick.color': '#898781',
    'axes.grid': False, 'legend.frameon': False,
})

# ---- 参考色板 (light) ----
C_SLOT = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'magenta': '#e87ba4'}
INK, SEC, MUT, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
# 顺序色带: 蓝(温度) / 橙(含水率)
_SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
             '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
_SEQ_ORANGE = ['#fbe3d2', '#f7c9a4', '#f2ac74', '#ed8f48', '#eb6834', '#d3521f', '#b94416',
               '#9c3808', '#7f2d05', '#632202']
cmap_T = LinearSegmentedColormap.from_list('seqT', _SEQ_BLUE)
cmap_C = LinearSegmentedColormap.from_list('seqC', _SEQ_ORANGE)
cmap_div = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])


def load(name):
    with open(os.path.join('figures', 'axisym_case_%s.json' % name), encoding='utf-8') as fh:
        return json.load(fh)


def line_style(name):
    return {'A': ('1D(现有模型)', C_SLOT['blue'], '-'),
            'B': ('β=0(端面绝热)', C_SLOT['orange'], '--'),
            'C05': ('β=0.5', C_SLOT['aqua'], '-.'),
            'C1': ('β=1', C_SLOT['magenta'], ':'),
            'C2': ('β=2', C_SLOT['orange'], (0, (3, 1, 1, 1)))}[name]


def _axis(ax, xlab, ylab, xlim=None, ylim=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(SURF)
    ax.set_xlabel(xlab, color=SEC)
    ax.set_ylabel(ylab, color=SEC)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(True, color=GRID, lw=0.5)


def main():
    A = load('A')
    cases = {k: load(k) for k in ('B', 'C1', 'C05', 'C2')}
    T1 = np.array(A['T_out_series'])
    C1s = np.array(A['C_out_series'])
    t_axis = np.arange(1, 1801)
    out_r = np.arange(0.0, 2.001, 0.1)

    # ===== 汇总: 回归自检 + 误差表 =====
    summary = {'t_report': [100, 300, 600, 900, 1200, 1500, 1800], 'cases': {}}
    for k, d in cases.items():
        Tm = np.array(d['T_mid_series']); Cm = np.array(d['C_mid_series'])
        Te = np.array(d['T_end_series']); Ce = np.array(d['C_end_series'])
        E_T_mid = np.max(np.abs(Tm - T1), axis=1)
        E_C_mid = np.max(np.abs(Cm - C1s), axis=1)
        E_T_end = np.max(np.abs(Te - T1), axis=1)
        E_C_end = np.max(np.abs(Ce - C1s), axis=1)
        summary['cases'][k] = {
            'E_T_mid': [float(E_T_mid[t - 1]) for t in summary['t_report']],
            'E_C_mid': [float(E_C_mid[t - 1]) for t in summary['t_report']],
            'E_T_end': [float(E_T_end[t - 1]) for t in summary['t_report']],
            'E_C_end': [float(E_C_end[t - 1]) for t in summary['t_report']],
            'max_over_t': {'E_T_mid': float(E_T_mid.max()), 'E_C_mid': float(E_C_mid.max()),
                           'E_T_end': float(E_T_end.max()), 'E_C_end': float(E_C_end.max())},
            'd1800': {kk: d['compare'][kk] for kk in
                      ('dT_1800_global_max', 'dC_1800_global_max', 'dT_1800_end_max', 'dC_1800_end_max')},
            'walltime_s': d['meta']['walltime_s'],
        }
        if k == 'B':
            summary['regression_B_vs_A'] = {
                'max_dT_all_t': float(np.max(np.abs(Tm - T1))),
                'max_dC_all_t': float(np.max(np.abs(Cm - C1s))),
            }
    # 网格收敛: conv50 / C1(100x50) / conv200 的中截面与端面 1800s 剖面
    conv = {}
    if os.path.exists('figures/axisym_case_conv50.json') and os.path.exists('figures/axisym_case_conv200.json'):
        g50 = load('conv50'); g200 = load('conv200')
        c1 = cases['C1']
        def reldiff(a, b): return float(np.max(np.abs(np.array(a) - np.array(b)) / (np.abs(np.array(b)) + 1e-12)))
        conv = {
            'T_mid_100vs50': reldiff(c1['T_mid_1800'], g50['T_mid_1800']),
            'T_mid_200vs100': reldiff(g200['T_mid_1800'], c1['T_mid_1800']),
            'C_mid_100vs50': reldiff(c1['C_mid_1800'], g50['C_mid_1800']),
            'C_mid_200vs100': reldiff(g200['C_mid_1800'], c1['C_mid_1800']),
            'T_end_100vs50': reldiff(c1['T_end_1800'], g50['T_end_1800']),
            'T_end_200vs100': reldiff(g200['T_end_1800'], c1['T_end_1800']),
            'C_end_100vs50': reldiff(c1['C_end_1800'], g50['C_end_1800']),
            'C_end_200vs100': reldiff(g200['C_end_1800'], c1['C_end_1800']),
            'corner_T': {'50x25': g50['T_end_1800'][-1], '100x50': c1['T_end_1800'][-1],
                         '200x100': g200['T_end_1800'][-1]},
            'corner_C': {'50x25': g50['C_end_1800'][-1], '100x50': c1['C_end_1800'][-1],
                         '200x100': g200['C_end_1800'][-1]},
        }
        summary['convergence'] = conv
    with open('figures/axial_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print('[axial_plots] 汇总已保存 figures/axial_summary.json')
    for k, v in summary['cases'].items():
        print('  [%s] E_T_mid(1800)=%.3e E_T_end(1800)=%.3e E_C_mid(1800)=%.3e E_C_end(1800)=%.3e walltime=%.0fs'
              % (k, v['E_T_mid'][-1], v['E_T_end'][-1], v['E_C_mid'][-1], v['E_C_end'][-1], v['walltime_s']))
    if 'regression_B_vs_A' in summary:
        print('  [回归] B vs A 全程最大差: dT=%.3e dC=%.3e' % (summary['regression_B_vs_A']['max_dT_all_t'],
                                                             summary['regression_B_vs_A']['max_dC_all_t']))
    if conv:
        print('  [收敛] T_mid 200vs100=%.2e, C_mid 200vs100=%.2e, T_end 200vs100=%.2e, C_end 200vs100=%.2e'
              % (conv['T_mid_200vs100'], conv['C_mid_200vs100'], conv['T_end_200vs100'], conv['C_end_200vs100']))

    # ===== 图1: 1800s r-z 场与偏差 (C1) =====
    c1 = cases['C1']
    rc = np.array(c1['fields_rc_cm']) / 100.0
    zc = np.array(c1['fields_zc_cm']) / 100.0
    T2 = np.array(c1['fields_1800_T'])
    C2 = np.array(c1['fields_1800_C'])
    T1_at = np.interp(rc, out_r / 100.0, T1[-1])
    C1_at = np.interp(rc, out_r / 100.0, C1s[-1])
    dT = T2 - T1_at[:, None]
    dC = C2 - C1_at[:, None]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 8.4), sharex=True)
    ext = [0, 2.0, 0, 12.5]
    im1 = axes[0, 0].imshow(T2.T, origin='lower', extent=ext, aspect='auto', cmap=cmap_T,
                            vmin=T2.min(), vmax=T2.max())
    axes[0, 0].set_title('(a) T(r,z) 1800s [℃]', color=INK, fontsize=9)
    im2 = axes[1, 0].imshow(dT.T, origin='lower', extent=ext, aspect='auto', cmap=cmap_div,
                            vmin=-np.abs(dT).max(), vmax=np.abs(dT).max())
    axes[1, 0].set_title('(b) ΔT = T(2D)−T(1D) [℃]', color=INK, fontsize=9)
    im3 = axes[0, 1].imshow(C2.T, origin='lower', extent=ext, aspect='auto', cmap=cmap_C,
                            vmin=C2.min(), vmax=C2.max())
    axes[0, 1].set_title('(c) C(r,z) 1800s [kg/kg]', color=INK, fontsize=9)
    im4 = axes[1, 1].imshow(dC.T, origin='lower', extent=ext, aspect='auto', cmap=cmap_div,
                            vmin=-np.abs(dC).max(), vmax=np.abs(dC).max())
    axes[1, 1].set_title('(d) ΔC = C(2D)−C(1D) [kg/kg]', color=INK, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('r [cm]')
    for ax in axes[:, 0]:
        ax.set_ylabel('z [cm]')
    for ax in axes.ravel():
        _axis(ax, '', '')
        ax.tick_params(colors=MUT)
    for im, ax in ((im1, axes[0, 0]), (im2, axes[1, 0]), (im3, axes[0, 1]), (im4, axes[1, 1])):
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.18, fraction=0.05)
        cb.ax.tick_params(labelsize=7, colors=MUT)
    fig.tight_layout(h_pad=2.2)
    fig.savefig('figures/fig1_axial_field_1800.png', bbox_inches='tight')
    plt.close(fig)

    # ===== 图2: 中截面 z=0 径向剖面 (1D vs β) 900/1800s =====
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for tt, row in ((900, 0), (1800, 1)):
        for q, col in (('T', 0), ('C', 1)):
            ax = axes[row, col]
            key = 'T_mid_series' if q == 'T' else 'C_mid_series'
            Akey = 'T_out_series' if q == 'T' else 'C_out_series'
            labels = []
            for name, (lab, colr, ls) in {'A': line_style('A'), 'C05': line_style('C05'),
                                          'C1': line_style('C1'), 'C2': line_style('C2')}.items():
                yy = np.array(A[Akey])[tt - 1] if name == 'A' else np.array(cases[name][key])[tt - 1]
                ax.plot(out_r, yy, color=colr, ls=ls, lw=2, label=lab)
                labels.append(lab)
            ax.set_title('%s(r)  t=%ds,  z=0 中截面' % ('温度' if q == 'T' else '含水率', tt),
                         color=INK, fontsize=9)
            _axis(ax, 'r [cm]', '℃' if q == 'T' else 'kg/kg')
            ax.legend(fontsize=7, loc='best', labelcolor=SEC)
    fig.tight_layout()
    fig.savefig('figures/fig2_midplane_profiles.png', bbox_inches='tight')
    plt.close(fig)

    # ===== 图3: 误差时间序列 (C1: 中截面 vs 端面) =====
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for q, ax in (('T', axes[0]), ('C', axes[1])):
        Tm = np.array(c1['T_mid_series'] if q == 'T' else c1['C_mid_series'])
        Te = np.array(c1['T_end_series'] if q == 'T' else c1['C_end_series'])
        Akey = 'T_out_series' if q == 'T' else 'C_out_series'
        E_mid = np.max(np.abs(Tm - np.array(A[Akey])), axis=1)
        E_end = np.max(np.abs(Te - np.array(A[Akey])), axis=1)
        ax.plot(t_axis, E_mid, color=C_SLOT['blue'], lw=2, label='中截面 z=0')
        ax.plot(t_axis, E_end, color=C_SLOT['orange'], lw=2, label='端面 z=12.5cm')
        ax.set_title('E_%s(t) = max_r |差值|  (β=1)' % ('T' if q == 'T' else 'C'), color=INK, fontsize=9)
        _axis(ax, 't [s]', 'Δ℃' if q == 'T' else 'Δ(kg/kg)')
        ax.legend(fontsize=7, labelcolor=SEC)
    fig.tight_layout()
    fig.savefig('figures/fig3_error_timeseries.png', bbox_inches='tight')
    plt.close(fig)

    # ===== 图4: 1800s 轴向分布 (C1: r=0 轴线 / r=R 表面) =====
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    zc_ = np.array(c1['prof_1800']['z_cm'])
    for q, ax in (('T', axes[0]), ('C', axes[1])):
        kax = 'T_axis_z' if q == 'T' else 'C_axis_z'
        ksf = 'T_surf_z' if q == 'T' else 'C_surf_z'
        ax.plot(zc_, c1['prof_1800'][kax], color=C_SLOT['blue'], lw=2, label='r=0 轴线')
        ax.plot(zc_, c1['prof_1800'][ksf], color=C_SLOT['orange'], lw=2, label='r=R 表面')
        ref = (T1 if q == 'T' else C1s)[-1]
        ax.axhline(ref[0], color=MUT, lw=1, ls='--')
        ax.axhline(ref[-1], color=MUT, lw=1, ls=':')
        ax.text(12.6, ref[0], '1D r=0', color=MUT, fontsize=7, va='center')
        ax.text(12.6, ref[-1], '1D r=R', color=MUT, fontsize=7, va='center')
        ax.set_title('1800s 轴向分布 (β=1): %s' % ('温度' if q == 'T' else '含水率'), color=INK, fontsize=9)
        _axis(ax, 'z [cm]', '℃' if q == 'T' else 'kg/kg', xlim=(0, 12.5))
        ax.legend(fontsize=7, loc='best', labelcolor=SEC)
    fig.tight_layout()
    fig.savefig('figures/fig4_axial_profiles.png', bbox_inches='tight')
    plt.close(fig)
    print('[axial_plots] 4 张图已保存 figures/')


if __name__ == '__main__':
    main()
