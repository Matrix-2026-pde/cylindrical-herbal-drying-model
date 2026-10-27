# 主程序：串联问题一求解 → 写 result1.xlsx → 灵敏度分析 → 汇总 all_results.json
# 用法（从工作区根）: "$PY" _utils/run_compute.py code/main.py
import os, sys, json, subprocess
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np

import problem1 as p1
import write_result as wr
import sensitivity_analysis as sa


def main():
    # 若 problem_1_results.json 缺失或需刷新，直接运行 problem1 主流程
    if not os.path.exists('figures/problem_1_results.json'):
        print("[main] 未见 problem_1 结果，运行 problem1.py ...")
        subprocess.run([sys.executable, os.path.join(_HERE, 'problem1.py')], check=True)

    p1res = json.load(open('figures/problem_1_results.json', encoding='utf-8'))
    wr.write()
    if not os.path.exists('figures/sensitivity_results.json'):
        sa.run()
    sens = json.load(open('figures/sensitivity_results.json', encoding='utf-8'))

    allres = {
        'problem_1': {
            'meta': p1res['meta'],
            'key_nodes': p1res['key_nodes'],
            'table_T': p1res['table_T'],
            'table_C': p1res['table_C'],
            'grid_convergence': p1res['grid_convergence'],
            'grid_conv_maxreldiff': p1res['grid_conv_maxreldiff'],
            'logic_probes': p1res['logic_probes'],
        },
        'sensitivity': sens,
        'note_problems_2_3_4': '缺上游 PROBLEM_ANALYSIS/MODELING 覆盖，本步不生成 result2/3/4；待后续阶段补齐',
    }
    with open('figures/all_results.json', 'w', encoding='utf-8') as fh:
        json.dump(allres, fh, ensure_ascii=False)
    print("[main] 已保存 figures/all_results.json")


if __name__ == '__main__':
    main()
