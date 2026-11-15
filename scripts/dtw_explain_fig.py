#!/usr/bin/env python3
"""dtw_explain_fig.py -- a schematic for the time-alignment (DTW) idea, in Chinese: the adult's clip and a robot doing
the same steps 25 % slower, compared frame by frame (today's cost) and step by step (aligned).
-> outputs/dtw/explain.png
usage: uv run python scripts/dtw_explain_fig.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parent.parent
FONT = Path.home() / ".local/share/fonts/NotoSansCJKtc-Regular.otf"
font_manager.fontManager.addfont(str(FONT))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()

T, N, REF_W, ROB_W = 300, 10, 30, 40                  # 300 frames; adult: 10 steps of 30 frames; robot: 40 frames each
CMAP = plt.get_cmap("viridis")
REF_Y, ROB_Y, H = 2.0, 0.4, 0.6
GOOD, BAD = "#2a9d4b", "#d62828"


def strip(ax, y, width, label):
    k = 0
    while k * width < T:
        x0, x1 = k * width, min((k + 1) * width, T)
        ax.add_patch(plt.Rectangle((x0, y), x1 - x0, H, color=CMAP(k / (N - 1)), ec="white", lw=2))
        if x1 - x0 > width * 0.6:
            ax.text((x0 + x1) / 2, y + H / 2, str(k + 1), ha="center", va="center", fontsize=17,
                    color="white" if k < 6 else "black")
        k += 1
    ax.text(-6, y + H / 2, label, ha="right", va="center", fontsize=15, multialignment="center")


def link(ax, x_rob, x_ref, ok, note):
    c = GOOD if ok else BAD
    ax.annotate("", xy=(x_ref, REF_Y), xytext=(x_rob, ROB_Y + H),
                arrowprops=dict(arrowstyle="<->", color=c, lw=2.2, shrinkA=0, shrinkB=0))
    xm = (x_rob + x_ref) / 2
    ax.text(xm, (REF_Y + ROB_Y + H) / 2, ("○ " if ok else "× ") + note, ha="center", va="center", fontsize=13,
            color=c, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=1.2))


def panel(ax, title, caption):
    ax.set_xlim(-95, T + 5); ax.set_ylim(-1.35, 3.3)
    ax.axis("off")
    strip(ax, REF_Y, REF_W, "大人\n（參考動作）")
    strip(ax, ROB_Y, ROB_W, "機器人\n（同樣動作，\n慢 25%）")
    for f in range(0, T + 1, 60):
        ax.plot([f, f], [ROB_Y - 0.06, ROB_Y - 0.12], color="0.4", lw=1)
        ax.text(f, ROB_Y - 0.2, f"{f // 30} 秒", ha="center", va="top", fontsize=11, color="0.35")
    ax.text(T / 2 - 45, 3.15, title, ha="center", va="center", fontsize=19, weight="bold")
    ax.text(T / 2 - 45, -0.85, caption, ha="center", va="center", fontsize=14, linespacing=1.5)


fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
fig.patch.set_facecolor("white")
panel(axes[0], "現在：拿「同一個時間點」來比",
      "時間越往後，機器人落後越多，\n明明是同一套動作，卻一直被判「不像」\n→ 搜尋只好逼它跟大人一樣快，跟不上就變形或摔倒")
for t in (15, 75, 135, 195, 255):
    ref, rob = t // REF_W + 1, t // ROB_W + 1
    link(axes[0], t, t, ref == rob, f"{ref} 比 {rob}")
panel(axes[1], "時間對齊：拿「同一個動作步驟」來比",
      "先找出「機器人現在做到大人的第幾步」，再比像不像\n→ 慢一點也不會被扣分\n300 幀內做到第 7.5 步 → 完成 75%（沒做完的部分另外扣分）")
for k in range(1, 8):
    link(axes[1], (k - 0.5) * ROB_W, (k - 0.5) * REF_W, True, f"{k} 比 {k}" if k in (1, 4, 7) else "")
for ax in axes:                                   # drop the empty boxes of unlabelled aligned links
    for t in list(ax.texts):
        if t.get_text() == "○ ":
            t.remove()
fig.text(0.5, 0.955, "數字 = 動作進行到第幾步（例如走路的第幾個姿勢）　　○ 比的是同一步　× 比的是不同步",
         ha="center", fontsize=14, color="0.3")
plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
out = REPO / "outputs/dtw/explain.png"; out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out); print(f"-> {out}")
