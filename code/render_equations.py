# -*- coding: utf-8 -*-
# 把"加 z 后的热传递/水分传递模型"渲染成 PNG
# 两遍式排版: 测量实际宽高(px/dpi) -> 超宽自动缩号 -> 英寸坐标(dpi_scale_trans)摆放 -> 断言无重叠无出界
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'SimHei', 'sans-serif'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm',
})

ORANGE = '#eb6834'
GRAY = '#52514e'
BLACK = '#0b0b0b'
DPI = 100.0

MAX_W = 11.4          # 文本最大宽度(英寸)
MARGIN = 0.5          # 四周边距(英寸)
LINE_GAP = 1.45       # 行距 = 实测行高 × LINE_GAP
SEC_GAP = 0.30        # 小节间额外间距(英寸)

LINES = [
    (r'第一问：加入轴向 $z$ 后的模型（圆柱坐标 · 轴对称 · 半域）', 16, BLACK, True, 0.0),
    (r'求解域  $\Omega=\{(r,z):\ 0\leq r\leq 0.02\ \mathrm{m},\quad 0\leq z\leq 0.125\ \mathrm{m}\}$   '
     r'（$z=0$ 中截面，$z=0.125$ m 端面）', 13, GRAY, False, 0.0),
    (r'① 热传递模型（温度 $T(r,z,t)$，℃）', 15, BLACK, True, SEC_GAP),
    (r'原一维径向方程（不变）：', 13, GRAY, False, 0.0),
    (r'$\rho\,c_p\,\dfrac{\partial T}{\partial t}\ =\ \dfrac{1}{r}\dfrac{\partial}{\partial r}\left(r\,k\,\dfrac{\partial T}{\partial r}\right)$', 20, BLACK, False, 0.0),
    (r'新增轴向项（变化点）：', 13, GRAY, False, 0.0),
    (r'$+\ \dfrac{\partial}{\partial z}\left(k\,\dfrac{\partial T}{\partial z}\right)$', 20, ORANGE, False, 0.0),
    (r'初值 $T(r,z,0)=28$；边界 $r=0$: $\partial T/\partial r=0$；$r=R$: $-k\,\partial T/\partial r = h\,(T-T_\infty(t))$（均不变）', 13, BLACK, False, 0.0),
    (r'新增边界：$z=0$: $\partial T/\partial z=0$（对称）；$z=H$: $-k\,\partial T/\partial z = h_b\,(T-T_\infty(t))$，$h_b=\beta h$', 13, ORANGE, False, 0.0),
    (r'② 水分传递模型（干基含水率 $C(r,z,t)$，kg/kg）', 15, BLACK, True, SEC_GAP),
    (r'原一维径向方程（不变）：', 13, GRAY, False, 0.0),
    (r'$\dfrac{\partial C}{\partial t}\ =\ \dfrac{1}{r}\dfrac{\partial}{\partial r}\left(r\,D(C)\,\dfrac{\partial C}{\partial r}\right)$', 20, BLACK, False, 0.0),
    (r'新增轴向项（变化点）：', 13, GRAY, False, 0.0),
    (r'$+\ \dfrac{\partial}{\partial z}\left(D(C)\,\dfrac{\partial C}{\partial z}\right),\qquad D(C)=7\times10^{-9}\,e^{-0.89/C}$', 20, ORANGE, False, 0.0),
    (r'初值 $C(r,z,0)=2.55$；边界 $r=0$: $\partial C/\partial r=0$；$r=R$: $-D\,\partial C/\partial r = h_m\,(C-C_{eq}(t))$（均不变）', 13, BLACK, False, 0.0),
    (r'新增边界：$z=0$: $\partial C/\partial z=0$（对称）；$z=H$: $-D\,\partial C/\partial z = h_{mb}\,(C-C_{eq}(t))$，$h_{mb}=\beta h_m$', 13, ORANGE, False, 0.0),
    (r'③ 变化的本质', 14, BLACK, True, SEC_GAP),
    (r'每个方程只多一个轴向二阶项；物性 $\rho,c_p,k,h,h_m$、经验式 $D(C)$、初值、侧面边界、', 12.5, BLACK, False, 0.0),
    (r'时间积分方式（温度 Crank–Nicolson、水分 backward-Euler+Picard）全部沿用附录2口径不变。', 12.5, BLACK, False, 0.0),
    (r'④ 两点物理提醒', 14, BLACK, True, SEC_GAP),
    (r'(i) $C_z(r,0,t)=0$ 只说明中截面一阶导为零，不代表 $C_{zz}=0$，中截面仍受轴向扩散间接影响；', 12.5, BLACK, False, 0.0),
    (r'(ii) 第一问 $D$ 不含 $T$、热物性为常数，加 $z$ 不会自动引入热湿耦合，两方程仍解耦。', 12.5, BLACK, False, 0.0),
]

# ---- 第一遍: 测量(px -> 英寸, 除 dpi) ----
probe = plt.figure(figsize=(MAX_W + 2 * MARGIN, 25), dpi=DPI)
probe.canvas.draw()
rend = probe.canvas.get_renderer()
placed = []
y = MARGIN
for text, fs, color, bold, gap in LINES:
    y += gap
    while True:
        t = probe.text(0, 0, text, fontsize=fs)
        bb = t.get_window_extent(rend)
        w_in, h_in = bb.width / DPI, bb.height / DPI
        if w_in <= MAX_W or fs <= 9:
            break
        fs -= 0.5
    placed.append((text, fs, color, bold, y, h_in))
    y += h_in * LINE_GAP
total_h = max(pt[4] + pt[5] for pt in placed) + MARGIN * 0.8
plt.close(probe)

# ---- 第二遍: 基线绘制 -> 实测墨水顶 -> 精确定位到 y_top ----
fig = plt.figure(figsize=(MAX_W + 2 * MARGIN, total_h), dpi=DPI)
arts = []
for text, fs, color, bold, y_top, h in placed:
    art = fig.text(0, 0, text, fontsize=fs, color=color,
                   weight='bold' if bold else 'normal', va='baseline', ha='left',
                   transform=fig.dpi_scale_trans)
    arts.append((art, y_top))
fig.canvas.draw()
rend2 = fig.canvas.get_renderer()
for art, y_top in arts:
    bb = art.get_window_extent(rend2)
    dy_top = bb.y1 / DPI                                  # 墨水顶高于基线的距离(英寸)
    art.set_position((MARGIN, total_h - y_top - dy_top))  # 墨水顶精确落在 y_top
fig.canvas.draw()

# ---- 第三遍: 逐行校验(画布内 + 无重叠: 本行顶 <= 上行底) ----
W_px, H_px = fig.bbox.width, fig.bbox.height
prev_bottom = 1e9                                   # 上一行的墨水底(英寸)
for i, art in enumerate(fig.texts):
    bb = art.get_window_extent(rend2)
    x0, y0, x1, y1 = bb.x0 / DPI, bb.y0 / DPI, bb.x1 / DPI, bb.y1 / DPI
    assert x0 >= -0.02 and y0 >= -0.02 and x1 <= W_px / DPI + 0.02 and y1 <= H_px / DPI + 0.02, \
        '[出界] 第%d行 bbox=(%.2f,%.2f)-(%.2f,%.2f)' % (i, x0, y0, x1, y1)
    assert y1 <= prev_bottom + 0.03, \
        '[重叠] 第%d行: 本行顶 %.3f in > 上行底 %.3f in' % (i, y1, prev_bottom)
    prev_bottom = y0
fig.savefig('figures/第一问_轴向扩展模型公式.png', bbox_inches='tight',
            facecolor='white', pad_inches=0.05, dpi=200)
print('[check] %d 行全部在画布内且无重叠; 画布 %.2f x %.2f in' % (len(fig.texts), W_px / DPI, H_px / DPI))
for text, fs, _, _, y_top, h in placed:
    print('   y_top=%5.2f in  h=%4.2f in  fs=%4.1f  %s' % (y_top, h, fs, text[:34]))

