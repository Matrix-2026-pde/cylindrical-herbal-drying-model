# 数据读取验证：确认附件1结构、前31行区间、无缺失
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pandas as pd
import params as P

# 显式读全部 sheet，暴露多表情况（附件1单表，读全部再取 Sheet1）
_all = pd.read_excel(P.ANNEX1_PATH, sheet_name=None)
print("附件1 共 %d 个 sheet: %s" % (len(_all), ", ".join("%s(%d行)" % (n, len(d)) for n, d in _all.items())))

df = _all['Sheet1']
print("列名:", list(df.columns))
print("总行数:", len(df))
print("缺失值:\n", df.isnull().sum().to_dict())

sub = df.iloc[:P.ANNEX1_NROWS]
print("\n前31行(t=0..1800s):")
print("  时间: %s -> %s, 间隔=%s" % (sub['时间'].iloc[0], sub['时间'].iloc[-1], sub['时间'].iloc[1]-sub['时间'].iloc[0]))
print("  温度范围: %.4f -> %.4f" % (sub['温度'].min(), sub['温度'].max()))
print("  水分浓度范围: %.5f -> %.5f" % (sub['水分浓度'].min(), sub['水分浓度'].max()))
assert sub['时间'].iloc[0] == 0 and sub['时间'].iloc[-1] == 1800, "时间区间不符"
assert not sub[['时间','温度','水分浓度']].isnull().any().any(), "前31行存在缺失"
print("\n[data_check] 附件1 前31行完整、区间正确 OK")
