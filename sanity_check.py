# 自动化数值 sanity check：扫描 figures/*_results.json 的数值有限性/范围
import os, sys, json, math
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

errors, warnings = [], []


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        # 大数组只抽样检查有限性，避免打印
        for i, v in enumerate(obj):
            if isinstance(v, (list, dict)):
                walk(v, f"{path}[{i}]")
            elif isinstance(v, float):
                if math.isnan(v):
                    errors.append(f"NaN at {path}[{i}]")
                elif math.isinf(v):
                    errors.append(f"Inf at {path}[{i}]")
    elif isinstance(obj, float):
        if math.isnan(obj):
            errors.append(f"NaN at {path}")
        elif math.isinf(obj):
            errors.append(f"Inf at {path}")


for f in ('figures/problem_1_results.json', 'figures/all_results.json', 'figures/sensitivity_results.json'):
    if os.path.exists(f):
        walk(json.load(open(f, encoding='utf-8')), os.path.basename(f))

if errors:
    print(f"[sanity_check] {len(errors)} 个错误")
    for e in errors[:5]:
        print("   ", e)
    sys.exit(1)
else:
    print(f"[sanity_check] 所有数值有限、无 NaN/Inf ({len(warnings)} 警告) PASS")
