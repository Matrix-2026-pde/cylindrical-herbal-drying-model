# -*- coding: utf-8 -*-
# 论文第10节复算程序: 收敛性与模型空间（新求解器口径: 节点FVM + BDF + Gauss面系数）
# A: Q2 表值空间稳定性(n=320/640)
# B: Q3 空间收敛 n=200/400/800 + 时间rtol + 最大步长 + 求积阶（生产 n=1280 复用 problem_3_results.json）
# C: Q4 同矩阵（生产 n=1280 复用 problem_4_results.json）
# D: 模型空间变体（论文8.3: advect/dilution/fixed/dirichlet/环境尾/半径插值）
# E: 与外部独立实现的交叉对照汇总
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
from solve_q234 import (DryingCylinder, solve_fixed, solve_moving, C_THRESHOLD,
                        R0, T_INIT, C_INIT)
FIG = os.path.join(_HERE, 'figures')


def _tstar(problem, n, rtol, max_step, quad, **kw):
    if problem == 'q3':
        model = DryingCylinder(n=n, pset='a3', fixed_radius=True, env_tail='nominal',
                               quadrature=quad)
        summ, _ = model.run(rtol=rtol, max_step=max_step, horizon=200 * 3600.0,
                            stop_event=True, audit=False)
    else:
        model = DryingCylinder(n=n, pset='a4', fixed_radius=False, env_tail='nominal',
                               quadrature=quad, **kw)
        summ, _ = model.run(rtol=rtol, max_step=max_step, horizon=200 * 3600.0,
                            stop_event=True, audit=False)
    return summ['event_h']


def _variant(kw, n=640, rtol=1e-8):
    model = DryingCylinder(n=n, pset='a4', fixed_radius=False, quadrature=8, **kw)
    summ, _ = model.run(rtol=rtol, max_step=120.0, horizon=200 * 3600.0,
                        stop_event=True, audit=False)
    return summ['event_h']


def _worker(cfg):
    t0 = time.time()
    tag = cfg.pop('tag')
    val = _tstar(**cfg)
    return tag, val, time.time() - t0


def _worker_var(cfg):
    t0 = time.time()
    tag = cfg.pop('tag')
    kw = dict(cfg.pop('kw'))
    val = _variant(kw, n=cfg.get('n', 640), rtol=cfg.get('rtol', 1e-8))
    return tag, val, time.time() - t0


if __name__ == '__main__':
    os.chdir(_HERE)
    out = {}

    print('=' * 70, flush=True)
    print('[A] Q2 表值空间稳定性 (n=320/640, 3h 五位置)', flush=True)
    from multiprocessing import Pool
    for n in (320, 640):
        model = DryingCylinder(n=n, pset='a3', fixed_radius=True, env_tail='nominal')
        summ, arr = model.run(rtol=1e-8, max_step=120.0, horizon=3 * 3600.0,
                              stop_event=False, audit=False, output_dt=1.0)
        i3 = int(np.argmin(np.abs(arr['times'] - 3 * 3600.0)))
        row = arr['c'][i3]
        vals = [float(np.interp(r / 100 / R0, model.x, row)) for r in (0, .5, 1, 1.5, 2)]
        print('  n=%3d: 3h C = [%s]' % (n, ', '.join('%.4f' % v for v in vals)), flush=True)
        out['q2_n%d' % n] = vals

    print('=' * 70, flush=True)
    print('[B/C] Q3/Q4 空间收敛 (rtol=1e-8, q8, max_step=120s)', flush=True)
    sweeps = [
        dict(tag='q3_n200', problem='q3', n=200, rtol=1e-8, max_step=120.0, quad=8),
        dict(tag='q3_n400', problem='q3', n=400, rtol=1e-8, max_step=120.0, quad=8),
        dict(tag='q3_n800', problem='q3', n=800, rtol=1e-8, max_step=120.0, quad=8),
        dict(tag='q4_n200', problem='q4', n=200, rtol=1e-8, max_step=120.0, quad=8),
        dict(tag='q4_n400', problem='q4', n=400, rtol=1e-8, max_step=120.0, quad=8),
        dict(tag='q4_n800', problem='q4', n=800, rtol=1e-8, max_step=120.0, quad=8),
    ]
    with Pool(3) as pool:
        rows = pool.map(_worker, sweeps)
    for tag, val, sec in rows:
        print('  %-8s t*=%.6f h (%.1fs)' % (tag, val, sec), flush=True)
        out[tag] = val

    print('=' * 70, flush=True)
    print('[B/C] 时间容差/最大步长/求积阶 敏感性 (n=400)', flush=True)
    sweeps2 = [
        dict(tag='q3_rtol1e-7', problem='q3', n=400, rtol=1e-7, max_step=120.0, quad=8),
        dict(tag='q3_rtol1e-9', problem='q3', n=400, rtol=1e-9, max_step=120.0, quad=8),
        dict(tag='q3_ms300', problem='q3', n=400, rtol=1e-8, max_step=300.0, quad=8),
        dict(tag='q3_q4pt', problem='q3', n=400, rtol=1e-8, max_step=120.0, quad=4),
        dict(tag='q3_q16pt', problem='q3', n=400, rtol=1e-8, max_step=120.0, quad=16),
        dict(tag='q4_rtol1e-7', problem='q4', n=400, rtol=1e-7, max_step=120.0, quad=8),
        dict(tag='q4_rtol1e-9', problem='q4', n=400, rtol=1e-9, max_step=120.0, quad=8),
        dict(tag='q4_ms300', problem='q4', n=400, rtol=1e-8, max_step=300.0, quad=8),
        dict(tag='q4_q4pt', problem='q4', n=400, rtol=1e-8, max_step=120.0, quad=4),
        dict(tag='q4_q16pt', problem='q4', n=400, rtol=1e-8, max_step=120.0, quad=16),
    ]
    with Pool(3) as pool:
        rows2 = pool.map(_worker, sweeps2)
    for tag, val, sec in rows2:
        print('  %-12s t*=%.6f h (%.1fs)' % (tag, val, sec), flush=True)
        out[tag] = val

    print('=' * 70, flush=True)
    print('[D] 模型空间变体 (论文8.3; n=640, rtol=1e-8)', flush=True)
    varcfgs = [
        dict(tag='advect_last', kw=dict(advect_variant=True, env_tail='last')),
        dict(tag='advect_nominal', kw=dict(advect_variant=True, env_tail='nominal')),
        dict(tag='dilution_nominal', kw=dict(dilution_variant=True, env_tail='nominal')),
        dict(tag='correct_last', kw=dict(env_tail='last')),
        dict(tag='correct_mean', kw=dict(env_tail='mean')),
        dict(tag='dirichlet', kw=dict(bc_moisture='dirichlet', env_tail='nominal')),
    ]
    with Pool(3) as pool:
        rows3 = pool.map(_worker_var, varcfgs)
    for tag, val, sec in rows3:
        print('  %-16s t*=%.4f h (%.1fs)' % (tag, val, sec), flush=True)
        out[tag] = val
    # 固定域 a4（无收缩上界）与 pchip 半径插值
    mf = DryingCylinder(n=640, pset='a4', fixed_radius=True, env_tail='nominal')
    sf, _ = mf.run(rtol=1e-8, max_step=120.0, horizon=200 * 3600.0, stop_event=True, audit=False)
    print('  fixed_a4         t*=%.4f h' % sf['event_h'], flush=True)
    out['fixed_a4'] = sf['event_h']
    mp = DryingCylinder(n=640, pset='a4', fixed_radius=False, env_tail='nominal', radius_mode='pchip')
    sp, _ = mp.run(rtol=1e-8, max_step=120.0, horizon=200 * 3600.0, stop_event=True, audit=False)
    print('  pchip_radius     t*=%.4f h' % sp['event_h'], flush=True)
    out['pchip_radius'] = sp['event_h']

    print('=' * 70, flush=True)
    print('[E] 与外部独立实现对照 (外部收敛值: Q3=57.5394, Q4=51.0906, 表值四位小数)', flush=True)
    j3 = json.load(open(os.path.join(FIG, 'problem_3_results.json'), encoding='utf-8'))
    j4 = json.load(open(os.path.join(FIG, 'problem_4_results.json'), encoding='utf-8'))
    print('  生产(本实现 n=1280): Q3 t*=%.4f h (外部 57.5394) | Q4 t*=%.4f h (外部 51.0906)'
          % (j3['meta']['t_star_h'], j4['meta']['t_star_h']), flush=True)
    print('  水分收支残差: Q3=%.3e Q4=%.3e (外部 ~7e-6 量级)' %
          (j3['audit']['balance_residual'], j4['audit']['balance_residual']), flush=True)
    print('  能量收支相对残差: Q4=%.3e' % j4['audit'].get('energy_rel', float('nan')), flush=True)
    with open(os.path.join(FIG, 'convergence_new.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('[done] 全部落盘 figures/convergence_new.json', flush=True)

