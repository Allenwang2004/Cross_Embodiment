# 新增一具身體：從 β 到可訓練的 MJCF

這份是 runbook。照著跑就會得到四樣東西：

```
assets/robots/<body>/robot.xml                     身形，actuator 與 joint 都是 adult 原值
assets/robot_torque/<body>/robot_torque_full.xml   actuator + joint 都按實測力矩比調整
data/<body>/retargeting_motion/                    540 段重定向動作
data/<body>/infer_retargeting_z/                   540 段逐幀 z（每段 (T, 256)）
```

全部用現有 script，沒有新程式。以 `tall_slim` 為實例，數字都是實跑出來的。

---

## 兩層的分工

```
adult/robot.xml ──scale_robot.py──> <body>/robot.xml ──torque_aggregate_motion_k.py──> robot_torque_full.xml
                     β：身形                                    k：力矩需求比
```

| 層 | 改什麼 | 不改什麼 |
|---|---|---|
| `scale_robot.py` | body pos、geom fromto/pos/size、Pelvis 貼地高度 | **armature / damping / stiffness 完全不動**（繼承 adult） |
| `torque_aggregate_motion_k.py --joint-dynamics` | actuator 四項、armature、damping、stiffness | 幾何與質量 |

第一層刻意不碰關節係數。它以前會乘 `length³·girth²`，但那是「關節自己那段肢體」的慣量，而關節實際承擔的是整條遠端 subtree——跨 body group 時就錯了（Neck 歸在 torso 卻扛著 head，child preset 讓 head 變大而 torso 變小，因子差 5.7 倍）。而且下游再乘一次會變成兩個無關因子相疊。現在統一由第二層從 adult 重寫絕對值。

---

## Step 1 — 產生身形

`scale_robot.py` 只有 `adult` / `child` / `elderly` 三個 preset，其他身體要把 8 個軸全部打在 CLI 上：

```bash
uv run scripts/scale_robot.py --label tall_slim \
  --leg-scale 1.15 --arm-scale 1.12 --torso-scale 1.1 --head-scale 0.95 \
  --leg-girth 0.85 --arm-girth 0.85 --torso-girth 0.85 --head-girth 0.9 \
  --no-actuator-scale
```

`--no-actuator-scale` 是**必要的**，不是選項。第二層的 k 定義成「相對 adult 的力矩需求比」，`joint_targets()` 會 assert src 的 actuator 與 adult 完全相同；先在這裡縮過，k 就會疊在另一個縮放上，直接報錯。

### ⚠️ 這一步會踩到的坑：`split` 被清掉

`generate()` 整個覆寫 `parameter.json`，只寫 `label` + `scale_actuators` + 8 個軸。`"split"` 會不見，而 `model/bilevel/data.py:load_body` 會 assert 它與 `config.py` 一致——不補回去，之後訓練會炸。

```bash
uv run scripts/write_body_splits.py --robots assets/robots
```

新身體如果還沒進 `BilevelConfig.train_bodies` / `heldout_bodies`，會被標成 `unused`；要先改 `model/bilevel/config.py` 再跑這行。

### `skeleton.json` 什麼時候要重生

只有 **β 改變**時才要。它記的是 body world_pos 與 mass，跟關節係數無關。tall_slim 這次 β 沒動，實測 `world_pos` 最大誤差 `0.00e+00`，所以沿用即可。β 真的變了就跑：

```bash
uv run scripts/export_skeleton_json.py \
  --input assets/robots/<body>/robot.xml --output assets/robots/<body>/skeleton.json
```

---

## Step 2 — 重定向 540 段動作

### ⚠️ 必須逐 clip 呼叫，不能一次丟整個 origin_motion

`qpos_retarget.py` 的批次模式用 `rglob` 收檔，但輸出是**攤平**的（`output_dir / stem.npz`）。而下游 `torque_ratio_across_motions.py` 是 `origin.iterdir()` 找子目錄、再讀 `retarget/<motion>/*.npz`，需要**巢狀**結構。所以要 54 次呼叫：

```bash
for d in data/origin_motion/*/; do
  m=$(basename "$d")
  uv run scripts/qpos_retarget.py \
    --input_dir  "data/origin_motion/$m" \
    --output_dir "data/tall_slim/retargeting_motion/$m" \
    --source_skeleton_json assets/robots/adult/skeleton.json \
    --target_skeleton_json assets/robots/tall_slim/skeleton.json \
    --target_xml           assets/robots/tall_slim/robot.xml
done
```

約 1 秒/段，54 段一分鐘內跑完。驗收：

```bash
find data/<body>/retargeting_motion -maxdepth 1 -mindepth 1 -type d | wc -l   # 要 54
find data/<body>/retargeting_motion -name '*.npz' | wc -l                     # 要 540
```

`--ground-correct` 預設開著，會逐幀把穿地的姿勢往上抬到 `--tolerance`（預設 -0.005 m）。tall_slim 因為比 adult 高（root height scale 1.1192），修正量比 child 小。

---

## Step 3 — 推論每具身體的 z

`batch_infer_z.py` 把 Step 2 的 qpos 灌進 HumEnv，讀 `get_obs()["proprio"]`，再跑 Metamotivo 的
`tracking_inference`，得到**逐幀**的 context vector。

```bash
uv run scripts/batch_infer_z.py \
  --input_dir  data/tall_slim/retargeting_motion \
  --output_dir data/tall_slim/infer_retargeting_z \
  --xml        assets/robots/tall_slim/robot.xml \
  --z0_dir     data/origin_z \
  --device     cuda
```

輸入 `<task>/<task>_<trial>.npz`（key `qpos`）→ 輸出 `<task>/<task>_<trial>.npy`，shape `(T, 256)`。
這支用 `rglob` + `relative_to`，**巢狀結構會自動保留**，不像 `qpos_retarget.py` 需要逐 clip 呼叫。

**`--xml` 必須指向目標身體。** 這是整步的重點：proprio 特徵要從**該身體自己的骨架**產生。指到
adult 就等於在問「adult 做這個動作時的 z」，那是 `data/infer_origin_z/`（同一支 script、
`--xml adult`、`--input_dir data/origin_motion`），不是這裡要的東西。

**`--z0_dir data/origin_z`** 會多寫一份 `cosine_summary.csv`（欄位 `clip,T,cos_mean,cos_min,cos_max`），
比對逐幀 z 與該段動作原本的 reward-inferred z0（shape `(1, 256)`）。這個數字說的是**換身體讓推論退化了多少**。

| | clips | cos_mean 中位 | 範圍 | cos_mean > 0.5 的 clip |
|---|---|---|---|---|
| child | 540 | 0.3571 | −0.044 – 0.919 | 190 |
| tall_slim | 540 | 0.4738 | −0.003 – 0.955 | 263 |

tall_slim 明顯較高，與它身形離 adult 較近一致。**這個數字低不代表跑錯**——它量的就是身體差異本身；
child 中位只有 0.36 是預期的。

> ⚠️ **`--device` 預設是 `cpu`。** 這裡用 `cuda`（540 段約一分半）。CPU 也會跑出結果，但如同
> `docs/scripts.md` 對執行緒數的警告，浮點歸約順序不同會有 ~1e-5 的漂移。要跨身體比較 z
> （例如擬合 z map）時，**同一批身體請用同一個 device**。

### 這份資料餵給誰

`fit_cross_body_z_map.py` — 學一個線性映射 W，把 adult 的 z 轉成該身體能執行的 z：

```bash
uv run scripts/fit_cross_body_z_map.py --src infer_origin_z --dst tall_slim/infer_retargeting_z
```

`--src` / `--dst` 吃的是 `data/` 底下的相對路徑，身體名從子目錄名推出來
（`tall_slim/infer_retargeting_z` → `tall_slim`）。

> ⚠️ 它預設寫到 `outputs/fit_cross_body_z_map/`，**沒有身體維度**。要為第二具身體跑之前，
> 先用 `--out-dir` 分開，否則會蓋掉 child 的結果——跟 `torque_ratio_across_motions.py` 同一個坑。

---

## Step 4 — 量測力矩需求比 k

輸出目錄的慣例是 **`<mode>/<body>/`**：

```bash
uv run scripts/torque_ratio_across_motions.py \
  --origin    data/origin_motion \
  --retarget  data/tall_slim/retargeting_motion \
  --adult-xml assets/robots/adult/robot.xml \
  --child-xml assets/robots/tall_slim/robot.xml \
  --outdir    outputs/torque_ratio_across_motions/gravity/tall_slim
```

> `--child-xml` 是命名遺留，意思是「另一具身體」，不是只能放 child。

**一定要給 `--outdir`。** 不給的話預設是 `outputs/torque_ratio_across_motions/<mode>`，會直接蓋掉別具身體的結果。同理，`torque_aggregate_motion_k.py --matrix` 的預設也已經失效，一定要指到 `<mode>/<body>/`。

量的是 `--mode gravity`：`qfrc_bias` at qvel=0，**純重力力矩，不含接觸力**。`mj_inverse` 會把地面反作用力算進來，但重定向後的腳幾乎每一幀都在接觸，沒有乾淨的參考幀，所以預設排除。這也是為什麼這套數據無法用來裁決「站立相單腿承受全身體重」這類接觸主導的假設。

---

## Step 5 — 先確認 k 真的是身體性質（別跳過）

整條方法的前提是「k 由身體決定，不由動作決定」。每具新身體都要驗一次，看 `_ratios_aggregate.csv`：

```bash
uv run python - <<'PY'
import csv,numpy as np
r=list(csv.DictReader(open('outputs/torque_ratio_across_motions/gravity/tall_slim/_ratios_aggregate.csv')))
km=np.array([float(x['k_aggregate']) for x in r])
kp=np.array([float(x['k_predicted_subtree']) for x in r])
nm=np.array([int(x['n_motions']) for x in r])
real=(nm>0)&(~np.isclose(km,kp,rtol=1e-9))       # 排除回退到幾何預測的關節，否則是循環論證
q=km[real]/kp[real]
print(f"獨立擬合 {real.sum()}/69   實測/幾何 中位 {np.median(q):.4f}  範圍 {q.min():.3f}-{q.max():.3f}")
print(f"偏差 >5%: {(abs(q-1)>.05).sum()}   >10%: {(abs(q-1)>.10).sum()}")
PY
```

實跑結果：

| | k 中位 | 實測/幾何預測 | 偏差>5% | 跨 54 段離散度（leg / arm / torso） |
|---|---|---|---|---|
| child | 0.2134 | 0.9997 `[0.879, 1.089]` | 4/48 | 0.2% / 0.2% / **25.9%** |
| tall_slim | 0.7579 | 1.0012 `[0.988, 1.035]` | **0/48** | 0.4% / 0.6% / **3.1%** |

判讀：

- **實測 k 幾乎等於幾何預測**（`k_predicted_subtree`，即質量×力臂比；等比例縮放下就是 s⁴）。兩具身體的中位都是 1.00。
- **腿、臂、頭跨動作離散度 <1%** → k 確實是身體性質。
- **torso 鏈是唯一的例外**。child 的 25.9% 來自軀幹關節重力訊號太弱、擬合是雜訊（`Chest_y` 只有 19 段動作能用、單 clip R²=0.003）。tall_slim 的 k≈0.76 離 1 較近、訊號較強，所以只有 3.1%。**如果新身體的 torso 離散度也衝到 20% 以上，那幾個關節的 k 不要相信，靠 `--r2-min` 讓它回退到幾何預測。**

既然兩具身體都證實了這件事，**新身體其實可以跳過 Step 4**，直接用 `k_predicted_subtree`（兩個 MJCF 各跑一次 `mj_forward` 就有）。但 Step 2（重定向）和 Step 3（z 推論）訓練本身就要，省不掉。

---

## Step 6 — 產生 robot_torque_full.xml

```bash
uv run scripts/torque_aggregate_motion_k.py \
  --matrix outputs/torque_ratio_across_motions/gravity/tall_slim \
  --src    assets/robots/tall_slim/robot.xml \
  --out    assets/robot_torque/tall_slim/robot_torque_full.xml \
  --joint-dynamics
```

`--joint-dynamics` 套用實測版等比例律。三個係數都從 **adult** 讀基準值，不經過 src：

| 屬性 | 因子 | 對應的等比例指數 |
|---|---|---|
| `armature` | × `Ir`（實測 subtree 慣量比） | s⁵ |
| `damping` | × `√(k·Ir)` | √(stiffness×inertia) |
| `stiffness` | × `k`（重力負載比） | s⁴ |
| actuator `gainprm` / `biasprm[0,1]` / `forcerange` | × `k` | — |
| actuator `biasprm[2]`（速度項） | × `√(k·Ir)`，跟著 damping | — |

`biasprm[2]` 走不同倍率不會破壞 `ctrl ∈ [-1,1] ⟺ qpos ∈ jnt_range` 的恆等式，因為平衡角 `q* = -(gainprm[0]·ctrl + biasprm[0])/biasprm[1]` 裡沒有它。

### 驗收看這幾行

script 自己會印。tall_slim 的實際輸出：

```
armature/damping/stiffness matches the law: 0.00e+00
Ir (subtree inertia ratio): median 0.7395  range 0.4579-0.9654
zeta    vs reference: median 1.000  range 0.994-1.006
tau=C/K vs reference: median 0.986  range 0.806-1.089
omega_n vs reference: median 1.015  range 0.918-1.241
dt*sqrt(Kp/I) max 0.209        (顯式積分需 < 2)
left/right mirror holds for all 27 pairs
```

- **ζ ≈ 1.000 是代數恆等，不是實測發現。** `armature × Ir` 讓總慣量恰好 ∝ Ir，代進 `ζ = C/(2√(KI))` 就約掉了。這行驗證的是算術有沒有寫對，等同單元測試。偏離 1.000 的那 ~0.6% 來自鏡像對稱化（L/R 共用一個 Ir）。
- **τ 和 ω_n 才有實質內容**，它們沒被設計成任何值，是 Froude 縮放跑出來的：τ ≈ √s_eff、ω_n ≈ 1/√s_eff。tall_slim 的 s_eff 接近 1（比 adult 略高略瘦），所以兩者都貼近 1.0；child 的 s_eff≈0.69，τ=0.828、ω_n=1.208，時鐘快 21%。
- `dt√(Kp/I)` 必須 < 2。目前餘裕很大（0.209 / 2）。

**注意 ζ 保住的是「每個關節自己的值」，不是把大家拉到 1。** adult 本身 ζ 從 0.285（Spine_z）到 5.760（L_Thorax_x）跨 20 倍，軀幹刻意欠阻尼、手臂重度過阻尼；這個異質分布被原樣搬過去。

---

## 檔案落點速查

| 路徑 | 產生者 | 每具身體一份？ |
|---|---|---|
| `assets/robots/<body>/robot.xml` | `scale_robot.py` | ✔ |
| `assets/robots/<body>/parameter.json` | `scale_robot.py` + `write_body_splits.py` | ✔ |
| `assets/robots/<body>/skeleton.json` | `export_skeleton_json.py`（β 變才要重生） | ✔ |
| `data/<body>/retargeting_motion/<motion>/*.npz` | `qpos_retarget.py`（逐 clip） | ✔ |
| `data/<body>/infer_retargeting_z/<motion>/*.npy` | `batch_infer_z.py`（`--xml` 指目標身體） | ✔ |
| `data/<body>/infer_retargeting_z/cosine_summary.csv` | 同上，需給 `--z0_dir data/origin_z` | ✔ |
| `outputs/torque_ratio_across_motions/<mode>/<body>/` | `torque_ratio_across_motions.py`（**必給 `--outdir`**） | ✔ |
| `assets/robot_torque/<body>/robot_torque_full.xml` | `torque_aggregate_motion_k.py --joint-dynamics` | ✔ |

---

## 全部身體已對齊（2026-08-24）

`assets/robots/` 底下 11 具身體（adult 除外）都已用 `--no-actuator-scale` 重新產生一次，現在狀態一致：

- **actuator = adult 原值**（先前只有 `elderly` 是縮過的，gainprm 中位 0.8769，現已還原成 1.0）
- **armature / damping / stiffness = adult 原值**（先前每具都帶著各自的 `local_scale`）

重新產生前的 stiffness / adult 中位，留作對照：

```
child 0.1346  short_limbed 0.3144  petite 0.3277  teen 0.5152  elderly 0.5553
long_limbed 0.8555  pear_shaped 0.9025  tall_slim 1.0151  short_stocky 1.0648
athletic 1.3225  giant 2.5830
```

驗證過的不變量（11 具全部）：幾何、`body_pos`、質量逐位元不變；`skeleton.json`（adult / child /
tall_slim 有）對新 XML 的 `world_pos` 誤差 `0.00e+00`，不需重生。

重跑指令（β 從各自的 `parameter.json` 讀）：

```bash
for b in athletic child elderly giant long_limbed pear_shaped petite short_limbed short_stocky teen; do
  ARGS=$(uv run python -c "
import json
p=json.load(open('assets/robots/$b/parameter.json'))
print(' '.join(f'--{a.replace(\"_\",\"-\")} {p[a]}' for a in
  ['leg_scale','arm_scale','torso_scale','head_scale','leg_girth','arm_girth','torso_girth','head_girth']))")
  uv run scripts/scale_robot.py --label $b $ARGS --no-actuator-scale
done
uv run scripts/write_body_splits.py --robots assets/robots    # split 會被覆寫掉，一定要補
```

`adult` 不在清單裡。它是來源，`scale_robot.py` 不會改它的 XML，但 `--preset adult` 會覆寫
`parameter.json` 並清掉 `"split": "source"`。

### 下游影響

| 產物 | 受影響？ |
|---|---|
| `robot_torque_full.xml`（`--joint-dynamics`） | **否**。實測相對差 ≤ 1e-15 |
| `assets/robot_torque/child/robot_torque.xml` | **是，已過時** |
| `assets/robot_torque/child/robot_torque_move_only.xml` | **是，已過時** |
| 直接吃 `assets/robots/*/robot.xml` 的訓練 | **是** |

`robot_torque_full.xml` 免疫的原因：第二層讀 src 只為了取 subtree 慣量，算的是
`M[dof,dof] − armature`，寫進去的 armature 又被減掉，只剩浮點抵消的捨入誤差。

那兩個 `child/` 底下的舊檔仍帶著 stiffness `0.1346×`（照抄舊 src），而現在的 src 是 `1.0000×`。
它們代表的是「只縮 actuator」那條已被取代的路線，**沒有重新產生**——要用的話得先決定它們還算不算數。
