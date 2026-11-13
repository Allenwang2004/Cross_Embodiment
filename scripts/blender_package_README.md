# 論文圖的 Blender 渲染包

這個資料夾可以獨立運作，不需要專案的其他檔案。它會渲染 teaser 和方法圖裡的動作畫面（共 21 格），每格輸出一張 PNG。

## 內容

| 檔案 | 說明 |
|---|---|
| `render_blender.py` | 在 Blender 裡建立場景並渲染 |
| `scene.json` | 每一格、每個姿勢的幾何形狀、位置和顏色 |
| `meshes.npz` | G1 的網格 |

動作資料已經存在 `scene.json`，不會重新模擬。

## 需要什麼

- Blender（已測試 5.0.1；舊版本沒有測過）
- 有 GPU 會比較快，但不是必要

## 怎麼跑

先把資料夾解壓縮，在這個資料夾裡開終端機。

**1. 先試一格，確認能動：**

```bash
blender -b -P render_blender.py -- --panel teaser_g1 --samples 64
```

結果在 `renders/teaser_g1.png`。Windows 要寫 Blender 的完整路徑，例如 `"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"`；Mac 是 `/Applications/Blender.app/Contents/MacOS/Blender`。

**2. 全部渲染：**

```bash
blender -b -P render_blender.py --
```

**時間參考：** 不含 `method_first_*` 的 12 格、每格 128 samples，在 16 個 CPU 執行緒上約 1.5 分鐘。預設是 256 samples，時間大約加倍。

## 選項（寫在 `--` 後面）

| 選項 | 說明 | 預設 |
|---|---|---|
| `--panel 名稱` | 只渲染指定的格，可以重複 | 全部 |
| `--samples N` | 越高越乾淨，也越慢 | 256 |
| `--scale X` | 解析度倍數，`2` 就是兩倍 | 1 |
| `--device CPU` 或 `GPU` | 指定裝置 | 有 GPU 就用 GPU |
| `--threads N` | CPU 執行緒數 | 全部 |
| `--out 資料夾` | 輸出位置 | `renders` |

## 21 格的名稱

| 名稱 | 內容 |
|---|---|
| `teaser_adult` | Origin |
| `teaser_payload` | Payload（+38 kg，球是畫上去的） |
| `teaser_joint` | Restricted joint（紅色右腿） |
| `teaser_child` | Reshaped body |
| `teaser_g1` | Robot（G1） |
| `teaser_band` | Teaser 橫幅版：六種條件排成一列（白底） |
| `teaser_band_grey` | 同上（灰底） |
| `method_fall_child` | 跌倒：Reshaped body（白底） |
| `method_fall_g1` | 跌倒：G1（白底） |
| `method_adult` | Origin reference（跑步，白底） |
| `method_child` | Result：Reshaped body（跑步，白底） |
| `method_g1` | Result：G1（跑步，白底） |
| `method_first_adult_60` 等 9 格 | 第一版版面的身體（跑步）：`adult`、`child`、`g1` 各 3 幀（60、165、270 附近腳著地的幀），透明背景，只留影子 |

## 想調整畫面

打開 `render_blender.py`，改最上面 `the look` 那一段：

| 變數 | 影響 |
|---|---|
| `COLORS` | 身體、紅腿、重物的顏色（sRGB，0–1） |
| `FLOOR_COLOR`、`FLOOR_ROUGHNESS` | 地面的亮度；粗糙度越低，反光越明顯 |
| `FLOOR_LIFT` | 只影響白底的格：讓地面達到純白；如果背景沒有全白就調高 |
| `BODY_ROUGHNESS` | 身體的光澤 |
| `WORLD_STRENGTH` | 環境光；越低，零件之間的暗部越深 |
| `KEY_ENERGY`、`KEY_ANGLE`、`KEY_POS` | 主光的強度、影子的柔和度、影子的方向 |

## 渲染好之後

把 `renders/` 裡的 PNG 複製回專案的 `docs/figures/blender_renders/`，再執行：

```bash
uv run python scripts/plot_teaser.py draw
uv run python scripts/plot_teaser.py band
uv run python scripts/plot_teaser.py band_grey
uv run python scripts/plot_method_figure.py draw
uv run python scripts/plot_method_figure.py first
```

圖會自動改用這些渲染結果。檔名不要改。

動作資料有更新時（例如換成最終的搜尋結果），要先在專案裡執行 `uv run python scripts/export_blender_package.py`，重新產生這個資料夾。
