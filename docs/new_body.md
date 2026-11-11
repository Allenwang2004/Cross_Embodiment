# 新增一具身體：從 β 到可訓練的 MJCF

這份是 runbook。一行指令就會得到這幾樣東西（`scripts/build_body.py`，2026-11-12 起；程式在 `crossenbodiment/bodybuild/`）：

```
assets/robots/<body>/robot.xml                          身形，actuator 與 joint 都是 adult 原值
assets/robots/<body>/parameter.json                     β 8 軸 + label（舊欄位）+ name / kind / parent / builder / edit
assets/robots/<body>/skeleton.json                      rest pose（重定向讀它的骨盆高度）
data/<body>/retargeting_motion/<task>/<stem>.npz        720 段重定向動作（54 個動作資料夾）
outputs/torque_ratio_across_motions/gravity/<body>/     力矩比 k 的矩陣與彙整表
assets/robots_torque/<body>/robot_torque_full.xml       actuator + joint 都按實測力矩比調整（搜尋用的 --xml）
```

```bash
# 等比例身體（make_scaled_bodies.py 那 8 具的規則：長度 x L、粗細 x L*W）
uv run python scripts/build_body.py scaled len0.8_wid1.2 --len 0.8 --wid 1.2
# 8 軸自己給（沒給的軸 = 1.0），或用 preset
uv run python scripts/build_body.py scaled tall_slim --leg-scale 1.15 --arm-scale 1.12 --torso-scale 1.1 --head-scale 0.95 \
  --leg-girth 0.85 --arm-girth 0.85 --torso-girth 0.85 --head-girth 0.9
uv run python scripts/build_body.py scaled child --preset child
# 一整組：make_scaled_bodies.py 的 8 具、make_grid_bodies.sh 的 64 具格子、make_morph_path.sh 的形態路徑
uv run python scripts/build_body.py scaled-set all --parallel 4
uv run python scripts/build_body.py grid --parallel 4 --keep-tasks outputs/morph_study/keep_tasks.txt
uv run python scripts/build_body.py morph m2s 8 0.65 0.68 1.0 1.0 1.0 1.0 1.0 1.0 --parallel 3
```

步驟（`BodyPipeline`，順序固定）：`shape → skeleton → retarget（每個動作資料夾一個工作）→ amass（選用）→ torque_ratio →
aggregate → physics → prune（選用）→ registry`。**輸出已存在的步驟會跳過**，所以中斷後重跑同一行就會接著做完；`--force`
重跑選到的步驟（改了 β 就要 `--force` 從 shape 重做）、`--steps a,b` 只跑其中幾步、`--dry-run` 只印會跑哪些步驟。
每次平行啟動前都會看 `free -g`，啟動後要留 ≥ 50 GB（`--keep-gb`），不夠就減少 worker 或等。log 在
`outputs/body_build/<body>.log`（台灣時間）。其他旗標：

| 旗標 | 作用 |
|---|---|
| `--root DIR` | 把 assets/、data/、outputs/ 都寫到 DIR 底下（輸入一律讀 repo 的 adult 與 data/origin_motion）；驗證用 |
| `--workers N` | 每具身體同時跑幾個重定向工作（預設 2；54 個資料夾 6 個 worker 約 10 秒） |
| `--parallel N` | 同時建幾具身體（每具一個 process） |
| `--motions a,b` | 只重定向這些動作資料夾（預設全部 54 個） |
| `--amass-clips FILE` | 另外重定向這些 AMASS clip → `data/<body>/amass_retargeting_motion/ACCAD/<clip>.npz`（len1.4_wid0.8 用的格式；child 的在 `amass_data/amass_retargeting_motion/ACCAD`） |
| `--keep-tasks FILE` | 力矩校準（用全部動作）之後，刪掉不在清單裡的任務的重定向（舊的 `KEEP_TASKS`） |
| `--register` | 身體不在 `configs/bodies.yaml` 時，把 entry 插進去（不加的話只寫到 `outputs/body_build/<body>.bodies.yaml`） |

成人尺寸的變體（質量、關節、力量的改動；不做力矩校準，模擬時直接用 shape XML），以及增益變體：

```bash
uv run python scripts/build_body.py heavy-limb --scale 80 --parts Wrist,Hand --clip move-ego-0-2/move-ego-0-2_4   # payload
uv run python scripts/build_body.py lock-joints --name adult_lockRleg_bent --lock "R_Hip_x=13,R_Knee_x=15,..." --solref 0.005 --clip ...
#   (lock_joint_body.py --user-lock --solref 0.005 --clip ... uses its LOCK_DEG table: the limited leg's locks)
uv run python scripts/build_body.py limit-joints --lock-body adult_lockRleg_bent --clip ...                       # limited leg
uv run python scripts/build_body.py elderly --clip move-ego-0-2/move-ego-0-2_4 [--stoop 40 --slow 2 --restrict joint]
uv run python scripts/build_body.py strength --name adult_s08_gear --scale 0.8 --what gear
uv run python scripts/build_body.py gains --src assets/robots_torque/child/robot_torque_full.xml --kp 0.8 --kd 1.2 \
  --out assets/robots_torque/child_kp0.8_kd1.2/robot_torque_full.xml                     # 或 --m 1.2（k x 1.2）
uv run python scripts/build_body.py gains --src assets/robots/g1/gains_perjoint.json --kp 0.8 --kd 0.8 \
  --out assets/robots/g1/gains_perjoint_kp0.8_kd0.8.json                                 # 機器人的 gains json
```

舊的入口都還在、旗標不變，現在是薄包裝：`new_body_pipeline.sh`、`make_grid_bodies.sh`、`make_morph_path.sh`、
`make_scaled_bodies.py` 印出並執行對應的 `build_body.py` 指令；`scale_robot.py`、`export_skeleton_json.py`、
`qpos_retarget.py`、`torque_ratio_across_motions.py`、`torque_aggregate_motion_k.py`、`scale_body_gains.py`、
`heavy_arm_body.py`、`lock_joint_body.py`、`limit_joint_body.py`、`elderly_body.py` 呼叫同一套類別。只有
`torque_scale_actuators.py`（單一 clip 的舊路線）移到了 `archive/scripts/`。

| 類別（`crossenbodiment/bodybuild/`） | 一個工作 | 取代 |
|---|---|---|
| `ScaledBody`、`HeavyLimb`、`LockJoints`、`LimitJoints`、`Elderly`、`ActuatorStrength`（`BodyBuilder`） | 讀 adult → 改 → 骨盆貼地 → robot.xml + parameter.json | scale_robot.py、heavy_arm / lock_joint / limit_joint / elderly_body.py、strength_rollout.py 的 build |
| `SameTopologyRetargeter`（`Retargeter`） | adult 的 clip → 這具身體的參考動作 | qpos_retarget.py |
| `TorqueCalibrator` | k 矩陣 → 彙整 k → robot_torque_full.xml + 驗收 | torque_ratio_across_motions.py、torque_aggregate_motion_k.py、torque_scale_actuators.py 的函式庫 |
| `GainScaler` | kp / kd 倍率（MJCF 或 gains json） | scale_body_gains.py、kpkd_test.py prepare |
| `BodyPipeline` | 上面的步驟、跳過已完成的、記憶體檢查、log | 四份流程副本 |

機器人（G1 / H1 / Apollo）拓撲不同，不走這條：參考動作是 `model/robot_map.py` 的 `RobotMap` 加
`scripts/robot_roundtrip.py`（`outputs/<robot>/roundtrip/<stem>/`），增益是 `scripts/robot_gain_scaling.py`，沒有改。

驗證（2026-11-12）：用新流程把 `len0.8_wid1.2` 與 child 重建到 worktree 的暫存目錄，robot.xml、skeleton.json、720 段重定向、
力矩比檔案逐位元相同；`len0.8_wid1.2` 的 robot_torque_full.xml 也相同。child 的 robot_torque_full.xml 是 08-24 從 child 舊的
robot.xml 產生的（下面「全部身體已對齊」那節），現在的程式（新舊都一樣）從現在的 robot.xml 重建出來文字排版不同、數值差
≤ 1.1e-14（相對），所以存著的那份沿用。見 `tests/golden/verify_bodies.py`、`tests/test_bodybuild.py`。

下面是每一步在做什麼、怎麼驗收（原本的逐步手冊；指令都還能單獨跑），數字都是實跑出來的，以 `tall_slim` 為例。

---

## 兩層的分工

```
adult/robot.xml ──ScaledBody──> <body>/robot.xml ──TorqueCalibrator──> robot_torque_full.xml
                     β：身形                                    k：力矩需求比
```

| 層 | 改什麼 | 不改什麼 |
|---|---|---|
| `ScaledBody`（`scale_robot.py`） | body pos、geom fromto/pos/size、Pelvis 貼地高度 | **armature / damping / stiffness 完全不動**（繼承 adult） |
| `TorqueCalibrator`（`torque_aggregate_motion_k.py --joint-dynamics`） | actuator 四項、armature、damping、stiffness | 幾何與質量 |

第一層刻意不碰關節係數。它以前會乘 `length³·girth²`，但那是「關節自己那段肢體」的慣量，而關節實際承擔的是整條遠端 subtree——跨 body group 時就錯了（Neck 歸在 torso 卻扛著 head，child preset 讓 head 變大而 torso 變小，因子差 5.7 倍）。而且下游再乘一次會變成兩個無關因子相疊。現在統一由第二層從 adult 重寫絕對值。

---

## Step 1 — 產生身形（`shape`、`skeleton`）

`build_body.py scaled` 等於舊的：

```bash
uv run scripts/scale_robot.py --label tall_slim \
  --leg-scale 1.15 --arm-scale 1.12 --torso-scale 1.1 --head-scale 0.95 \
  --leg-girth 0.85 --arm-girth 0.85 --torso-girth 0.85 --head-girth 0.9 \
  --no-actuator-scale
uv run scripts/export_skeleton_json.py \
  --input assets/robots/<body>/robot.xml --output assets/robots/<body>/skeleton.json
```

`--no-actuator-scale` 是**必要的**（`build_body.py` 預設就是如此）。第二層的 k 定義成「相對 adult 的力矩需求比」，
`joint_targets()` 會 assert src 的 actuator 與 adult 完全相同；先在這裡縮過，k 就會疊在另一個縮放上，直接報錯。

### `parameter.json`：舊欄位不動，`split` 不再被清掉

舊的 `scale_robot.py` 整個覆寫 `parameter.json`，只寫 `label` + `scale_actuators` + 8 個軸，`"split"` 會不見。現在
`ScaledBody` 照舊寫這些欄位（`model/dataset.py` 的 `load_beta` 讀 8 個軸；8 軸的順序只有一份：
`crossenbodiment/bodybuild/beta.py` 的 `BETA_AXES`），後面多加 `name`、`kind`、`parent`、`builder`、`edit`（例如 len / wid），
目的地或 repo 裡同名身體原有的 `"split"` 會保留。`split` 只有 `scripts/build_dataset.py`（manifest）會讀；新身體要分
train / test 時改 `model/bodies.py` 再跑：

```bash
uv run scripts/write_body_splits.py --robots assets/robots
```

（舊版這裡提到的 `model/bilevel/`（`load_body`、`BilevelConfig`）已經移除，不再適用。）

成人尺寸的變體以前直接複製 adult 的 `parameter.json` / `skeleton.json`；現在 `parameter.json` 保留 adult 的欄位再加上
上述欄位（`edit` 記錄改了什麼），`skeleton.json` 由新的 XML 匯出（例如 payload 的手比較重，質量就是它自己的）。

### `skeleton.json` 什麼時候要重生

只有 **β 改變**時才要。它記的是 body world_pos 與 mass，跟關節係數無關。tall_slim 這次 β 沒動，實測 `world_pos` 最大誤差
`0.00e+00`。`build_body.py` 在 skeleton.json 不存在時匯出；β 改了要 `--force --steps shape,skeleton,...` 從頭重做。

---

## Step 2 — 重定向全部動作（`retarget`）

`SameTopologyRetargeter`：關節角照抄，root 位置乘上兩份 skeleton.json 骨盆高度的比，再逐幀把穿地的姿勢往上抬到
-0.005 m（只抬不壓）。`build_body.py` 每個動作資料夾開一個工作，輸出是**巢狀**的
`data/<body>/retargeting_motion/<task>/<stem>.npz`（下游 `torque_ratio_across_motions.py` 要這個結構）。舊的
`qpos_retarget.py` 批次模式輸出是**攤平**的，所以以前要逐資料夾呼叫 54 次：

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

驗收（現在每具 54 個資料夾、720 段；舊紀錄裡的 540 是當時的動作數）：

```bash
find data/<body>/retargeting_motion -maxdepth 1 -mindepth 1 -type d | wc -l   # 要 54
find data/<body>/retargeting_motion -name '*.npz' | wc -l                     # 要 720（= data/origin_motion）
```

tall_slim 因為比 adult 高（root height scale 1.1192），修正量比 child 小。

---

## Step 3 — 推論每具身體的 z（不在 `build_body.py` 裡，需要時手動跑）

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

11 具身體全部跑完（每具都是 540 clips），按 cos_mean 中位排序：

| body | root height scale | cos_mean 中位 | cos_mean > 0.5 的 clip |
|---|---|---|---|
| pear_shaped | 0.9891 | 0.5318 | 284 |
| athletic | 1.0027 | 0.5258 | 283 |
| elderly | 0.8981 | 0.5203 | 276 |
| short_stocky | 0.8522 | 0.4999 | 270 |
| teen | 0.8347 | 0.4938 | 267 |
| tall_slim | 1.1192 | 0.4738 | 263 |
| petite | 0.7833 | 0.4657 | 257 |
| long_limbed | 1.1169 | 0.4628 | 260 |
| giant | 1.2216 | 0.4138 | 242 |
| short_limbed | 0.6539 | 0.4106 | 228 |
| child | 0.6110 | 0.3571 | 190 |

（root height scale = `target_h / source_h`，兩個 `skeleton.json` 的 root rest height 之比，
就是 `qpos_retarget.py` 開頭印的那個數字。）

**這個數字低不代表跑錯**——它量的就是身體差異本身。排序大致跟「離 adult 多遠」一致：
`pear_shaped` / `athletic` 幾乎同高同比例，落在 0.53；`child`（0.36）和 `short_limbed`（0.41）
是身形離 adult 最遠的兩具，也是最低的兩具。

> ⚠️ **`--device` 預設是 `cpu`。** 這裡用 `cuda`（540 段約一分半）。CPU 也會跑出結果，但如同
> 執行緒數不同時的情況，浮點歸約順序不同會有 ~1e-5 的漂移。要跨身體比較 z
> 時，**同一批身體請用同一個 device**。

### 這份資料餵給誰

`scripts/build_dataset.py` 把它寫進 manifest(`infer_origin_z`、`retarget_z` 欄位);
`scripts/test_track_z.py` 拿它當「逐幀 backward z」的對照組。原本吃這份資料的 z map 路線
(`fit_cross_body_z_map.py` 等)已在 2026-10-04 移除,見 git branch `snapshot/pre-cleanup-2026-10-04`。

---

## Step 4 — 量測力矩需求比 k（`torque_ratio`）

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

`build_body.py` 一律寫到 `<mode>/<body>/`。單獨跑舊指令時**一定要給 `--outdir`**，不給的話預設是 `outputs/torque_ratio_across_motions/<mode>`，會直接蓋掉別具身體的結果。同理，`torque_aggregate_motion_k.py --matrix` 的預設也已經失效，一定要指到 `<mode>/<body>/`。

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

> ⚠️ **這一步要排在 Step 6 之後跑**（`build_body.py` 在 physics 之後印這行：`k median ... | independent fits ... |
> measured/predicted median ...`）。 `_ratios_aggregate.csv` 是 Step 6 的
> `torque_aggregate_motion_k.py` 寫進 matrix 目錄的；Step 4 只產 `_k_matrix.csv` /
> `_r2_matrix.csv` / `_corr.*`。實際順序是 **4 → 6 → 5**。

11 具全部實跑：

| body | k 中位 | 獨立擬合 | 實測/幾何預測 | 偏差>5% | >10% | torso 鏈最大 CV |
|---|---|---|---|---|---|---|
| child | 0.2134 | 48/69 | 0.9997 `[0.879, 1.089]` | 4 | 1 | 118.7% (`Chest_y`) |
| petite | 0.4096 | 12/69 | 0.9976 `[0.925, 1.033]` | 1 | 0 | 37.3% (`Chest_y`) |
| elderly | 0.5182 | 48/69 | 1.0029 `[0.898, 1.048]` | 1 | 1 | 42.5% (`Chest_y`) |
| teen | 0.6197 | 48/69 | 0.9998 `[0.941, 1.026]` | 1 | 0 | 23.4% (`Chest_y`) |
| short_limbed | 0.6406 | 45/69 | 0.9988 `[0.894, 1.049]` | 1 | 1 | 36.9% (`Chest_y`) |
| long_limbed | 0.6479 | 48/69 | 1.0014 `[0.978, 1.047]` | 0 | 0 | 9.7% (`Spine_z`) |
| tall_slim | 0.7579 | 48/69 | 1.0012 `[0.988, 1.035]` | 0 | 0 | 7.8% (`Torso_x`) |
| pear_shaped | 0.8348 | 48/69 | 0.9999 `[0.912, 1.023]` | 1 | 0 | 25.0% (`Chest_y`) |
| athletic | 1.3812 | 48/69 | 0.9988 `[0.983, 1.106]` | 1 | 1 | 19.8% (`Chest_y`) |
| short_stocky | 1.4547 | 48/69 | 0.9987 `[0.960, 1.025]` | 0 | 0 | 15.4% (`Torso_x`) |
| giant | 1.9310 | 48/69 | 1.0007 `[0.978, 1.092]` | 1 | 0 | 17.9% (`Chest_y`) |

（torso CV 是 `torque_ratio_across_motions.py` 自己那個 `spread = std/|mean|` 統計量，取軀幹鏈上最差的
單一關節，不是群組平均——所以跟這份文件舊版的「群組離散度」欄不是同一個定義，數字不能直接對照。）

判讀：

- **實測 k 幾乎等於幾何預測**（`k_predicted_subtree`，即質量×力臂比；等比例縮放下就是 s⁴）。
  11 具的中位全部落在 **0.9976–1.0029**，最大偏離 0.3%。k 本身跨了 9 倍（0.21 到 1.93），
  這個結論在整個範圍上都成立。
- **偏差>5% 的關節每具最多 4 個（48 個獨立擬合中），>10% 最多 1 個。**
- **torso 鏈仍然是唯一的例外**，而且離散度跟 k 離 1 的距離相關：`child`（k=0.21）118.7%、
  `elderly` 42.5%、`petite` 37.3%，而 `tall_slim`（k=0.76）只有 7.8%。**torso CV 衝到 20% 以上時，
  那幾個關節的 k 不要相信，靠 `--r2-min` 讓它回退到幾何預測。**

### `petite` 的 12/69 不是資料品質問題

`petite` 的「獨立擬合」只有 12/69，其他身體都是 45–48。這**不是**擬合失敗——它的 R² 通過率
0.990 還比 child 的 0.971 高。原因是 `petite` 的 β 讓腿和臂兩組**長度與粗細同值**
（`leg_scale = leg_girth = arm_scale = arm_girth = 0.80`），那兩條鏈就是純等比例縮放，
重力力矩比精確等於 `0.80⁴ = 0.4096`，實測值**逐位元命中**幾何預測，`std_across_motions = 0.000000`
（47 段動作）。落在這條線上的 57 個關節是 24 腿 + 24 臂 + 9 軀幹/頭。

Step 5 的檢查式用 `~np.isclose(km, kp)` 排除「回退到幾何預測」的關節以免循環論證，但它同時也把
**完全吻合**的關節排掉了。所以 β 越接近等比例的身體，這個分母就越小。看 `n_motions` 才是判斷有沒有
真的回退的依據——11 具全部 `n_motions == 0` 的關節數是 **0**，沒有任何一個關節是真的沒資料。

> Step 4 只花 **4.5 秒**（不是原本預期的重活），而且 Step 6 的 `--matrix` 本來就要讀它的輸出，
> 所以**不要跳過**。順便還能拿到上面這張表的驗收數字。

---

## Step 6 — 產生 robot_torque_full.xml（`aggregate`、`physics`）

```bash
uv run scripts/torque_aggregate_motion_k.py \
  --matrix outputs/torque_ratio_across_motions/gravity/tall_slim \
  --src    assets/robots/tall_slim/robot.xml \
  --out    assets/robots_torque/tall_slim/robot_torque_full.xml \
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

| 路徑 | 產生者（`build_body.py` 的步驟 / 舊指令） | 每具身體一份？ |
|---|---|---|
| `assets/robots/<body>/robot.xml` | `shape`（`ScaledBody`；`scale_robot.py`） | ✔ |
| `assets/robots/<body>/parameter.json` | `shape`（保留 `split`）+ `write_body_splits.py` | ✔ |
| `assets/robots/<body>/skeleton.json` | `skeleton`（`export_skeleton_json.py`；β 變才要重生） | ✔ |
| `data/<body>/retargeting_motion/<motion>/*.npz` | `retarget`（`qpos_retarget.py` 逐資料夾） | ✔ |
| `data/<body>/amass_retargeting_motion/ACCAD/*.npz` | `amass`（`--amass-clips`） | 選用 |
| `outputs/body_build/<body>.log` | `build_body.py` | ✔ |
| `data/<body>/infer_retargeting_z/<motion>/*.npy` | `batch_infer_z.py`（`--xml` 指目標身體） | ✔ |
| `data/<body>/infer_retargeting_z/cosine_summary.csv` | 同上，需給 `--z0_dir data/origin_z` | ✔ |
| `outputs/torque_ratio_across_motions/<mode>/<body>/` | `torque_ratio`、`aggregate`（`torque_ratio_across_motions.py` **必給 `--outdir`**） | ✔ |
| `assets/robots_torque/<body>/robot_torque_full.xml` | `physics`（`torque_aggregate_motion_k.py --joint-dynamics`） | ✔ |

---

## 歷史紀錄

以下兩節是 2026-08-24 的紀錄。表裡的 11 具舊身體（athletic、giant、tall_slim……）在 2026-11-11 刪掉了
（`outputs/body_shapes_deleted_2026-11-11.txt`），現在的身體是 child 與 8 具 len × wid（`configs/bodies.yaml`）。
當時的指令寫 `scale_robot.py` + `write_body_splits.py`；現在用 `build_body.py scaled <body> --force --steps shape` 重產身形
（`split` 會保留，不必再補）。

### 全部身體已對齊（2026-08-24）

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

#### 下游影響

| 產物 | 受影響？ |
|---|---|
| `robot_torque_full.xml`（`--joint-dynamics`） | **否**。實測相對差 ≤ 1e-15 |
| `assets/robots_torque/child/robots_torque.xml` | **是，已過時** |
| `assets/robots_torque/child/robots_torque_move_only.xml` | **是，已過時** |
| 直接吃 `assets/robots/*/robot.xml` 的訓練 | **是** |

`robot_torque_full.xml` 免疫的原因：第二層讀 src 只為了取 subtree 慣量，算的是
`M[dof,dof] − armature`，寫進去的 armature 又被減掉，只剩浮點抵消的捨入誤差。

那兩個 `child/` 底下的舊檔仍帶著 stiffness `0.1346×`（照抄舊 src），而現在的 src 是 `1.0000×`。
它們代表的是「只縮 actuator」那條已被取代的路線，**沒有重新產生**——要用的話得先決定它們還算不算數。

---

### 11 具身體全部跑完 Step 1–6（2026-08-24）

在此之前只有 `child` 和 `tall_slim` 走過這條流程。其餘 9 具
（`athletic` `elderly` `giant` `long_limbed` `pear_shaped` `petite` `short_limbed`
`short_stocky` `teen`）已補齊，現在 `assets/robots/` 底下 adult 以外的 11 具狀態一致：

| 產物 | 狀態 |
|---|---|
| `assets/robots/<body>/skeleton.json` | 11/11 |
| `data/<body>/retargeting_motion/` | 11/11，每具 54 dirs / 540 npz |
| `data/<body>/infer_retargeting_z/` | 11/11，每具 540 npy + `cosine_summary.csv` |
| `outputs/torque_ratio_across_motions/gravity/<body>/` | 11/11 |
| `assets/robots_torque/<body>/robot_torque_full.xml` | 11/11 |

Step 6 的驗收 9 具全綠：`armature`/`damping`/`stiffness` 對法則誤差 `0.00e+00`、
ζ 中位 1.000、27 組鏡像全對、`dt·√(Kp/I)` 最大 0.234（需 < 2）。
Froude 時鐘照預期跟著 s_eff 走：

| body | k 中位 | Ir 中位 | τ 中位 | ω_n 中位 | dt·√(Kp/I) |
|---|---|---|---|---|---|
| petite | 0.4096 | 0.3277 | 0.894 | 1.118 | 0.204 |
| elderly | 0.5182 | 0.4553 | 0.923 | 1.083 | 0.210 |
| teen | 0.6197 | 0.5501 | 0.942 | 1.061 | 0.190 |
| short_limbed | 0.6406 | 0.6202 | 0.984 | 1.016 | 0.162 |
| long_limbed | 0.6479 | 0.6074 | 0.973 | 1.027 | 0.234 |
| pear_shaped | 0.8348 | 0.7229 | 0.970 | 1.031 | 0.180 |
| athletic | 1.3812 | 1.6013 | 1.045 | 0.957 | 0.171 |
| short_stocky | 1.4547 | 1.3680 | 1.010 | 0.990 | 0.147 |
| giant | 1.9310 | 2.2514 | 1.086 | 0.921 | 0.173 |

`giant` 的 τ=1.086 / ω_n=0.921（時鐘比 adult 慢 9%）與 `petite` 的 0.894 / 1.118（快 12%）
是兩端，都是 √s_eff 直接跑出來的，沒有被設計成任何值。

#### 這批資料餵給誰

`scripts/build_dataset.py` 把 `data/<body>/retargeting_motion` 與 `assets/robots/<body>/parameter.json`
組成 `datasets/crossenbodiment-10bodies`（540 clips × 10 bodies = 5400 列，8 train / 2 test）。
它用 symlink 而不是複製，所以那 2.9 GB 只存在一份。`model/simple/train.py` 直接吃這個 manifest，
每個 update 抽一具身體。

#### 沒做的事

- **`data/<body>/ik_retargeting_action/` 與 `ik_retargeting_z/`**:只有 `child` 有。那是
  `ik_action_from_qpos.py` 與已移除的 `ik_z_from_action.py` 那條路線的產物,不在這份 runbook 的四樣
  交付物裡,所以沒有為其他 10 具產生。
