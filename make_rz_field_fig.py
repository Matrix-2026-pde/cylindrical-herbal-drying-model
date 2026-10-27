# -*- coding: utf-8 -*-
# 论文图8: 二维(r,z)轴对称场 1800s（温度/含水率热图），数据来自 problem1_axisym_results.json
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'SimHei', 'sans-serif'],
    'axes.unicode_minus': False, 'figure.dpi': 200, 'savefig.dpi': 200,
})
SEC, MUT, INK = '#52514e', '#898781', '#0b0b0b'
_SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
             '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
_SEQ_ORANGE = ['#fbe3d2', '#f7c9a4', '#f2ac74', '#ed8f48', '#eb6834', '#d3521f', '#b94416',
               '#9c3808', '#7f2d05', '#632202']
cmapT = LinearSegmentedColormap.from_list('t', _SEQ_BLUE)
cmapC = LinearSegmentedColormap.from_list('c', _SEQ_ORANGE)

d = json.load(open('figures/problem1_axisym_results.json', encoding='utf-8'))
f = d['field_1800_C1']
T = np.array(f['T']); C = np.array(f['C'])
rc = np.array(f['r_cm']); zc = np.array(f['z_cm'])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
im1 = axes[0].imshow(T.T, origin='lower', extent=[0, 2, 0, 12.5], aspect='auto',
                     cmap=cmapT, vmin=T.min(), vmax=T.max())
cb1 = fig.colorbar(im1, ax=axes[0], pad=0.02)
cb1.set_label('T / ℃', color=SEC); cb1.ax.tick_params(colors=MUT, labelsize=7)
im2 = axes[1].imshow(C.T, origin='lower', extent=[0, 2, 0, 12.5], aspect='auto',
                     cmap=cmapC, vmin=1.5, vmax=2.55)
cb2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
cb2.set_label('C / (kg·kg⁻¹)', color=SEC); cb2.ax.tick_params(colors=MUT, labelsize=7)
for ax in axes:
    ax.set_xlabel('r / cm', color=SEC)
    ax.tick_params(colors=MUT, labelsize=7)
    ax.set_facecolor('#fcfcfb')
axes[0].set_ylabel('z / cm', color=SEC)
axes[0].set_title('(a) T(r,z) 1800 s', fontsize=9, color=INK)
axes[1].set_title('(b) C(r,z) 1800 s', fontsize=9, color=INK)
for ax in axes:
    ax.axvline(2.0, color=MUT, lw=0.5)
fig.tight_layout(w_pad=1.2)
fig.savefig('paper_figs/fig8_rz_field.png', bbox_inches='tight')
print('[rz_field] saved')
