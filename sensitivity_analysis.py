# 灵敏度分析：h(±10%)、hm(±10%)、Ceq口径(±20%)、界面D平均方式 对中心/表面 T,C 的影响
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import problem1 as p1
from params import H_CONV, HM_MASS


def run():
    base = p1.solve()
    keys = ['T_center_1800', 'T_surface_1800', 'C_center_1800', 'C_surface_1800']
    baseline = {k: base[k] for k in keys}

    out = {'baseline': baseline, 'parameters': {}}

    # h ±10%
    hv = [H_CONV * f for f in (0.9, 1.0, 1.1)]
    out['parameters']['h_conv'] = {
        'values': hv,
        **{k: [p1.solve(h_conv=h)[k] for h in hv] for k in keys}}

    # hm ±10%
    hmv = [HM_MASS * f for f in (0.9, 1.0, 1.1)]
    out['parameters']['hm_mass'] = {
        'values': hmv,
        **{k: [p1.solve(hm=hm)[k] for hm in hmv] for k in keys}}

    # Ceq 口径 ±20%（核心软肋）
    cs = [0.8, 1.0, 1.2]
    out['parameters']['ceq_scale'] = {
        'values': cs,
        **{k: [p1.solve(ceq_scale=s)[k] for s in cs] for k in keys}}

    # 界面D平均方式 调和 vs 算术
    r_arith = p1.solve(d_interface='arithmetic')
    out['parameters']['d_interface'] = {
        'values': ['harmonic', 'arithmetic'],
        **{k: [base[k], r_arith[k]] for k in keys}}

    with open('figures/sensitivity_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("[sensitivity] 已保存 figures/sensitivity_results.json")
    for pname, pd in out['parameters'].items():
        print("  %s: values=%s" % (pname, pd['values']))
        print("     C_surface_1800=%s" % [round(x, 4) for x in pd['C_surface_1800']])
        print("     C_center_1800=%s" % [round(x, 4) for x in pd['C_center_1800']])
    return out


if __name__ == '__main__':
    run()
