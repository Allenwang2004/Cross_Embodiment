# For the old render strobe stuff
1. Edit stuff by default in render_strobe.yaml
2. Run the figure through this
```
MUJOCO_GL=egl .venv/bin/python render_strobe.py
```
3. To save the strobe yaml
```
MUJOCO_GL=egl .venv/bin/python render_strobe.py --save-config configs/strobe/jump-2_strobe.yaml
```
4. When you want to run it again
```
MUJOCO_GL=egl .venv/bin/python render_strobe.py --config configs/strobe/jump-2_strobe.yaml
```

# For the new trajectories replay strobe
1. edit configs/render_strobe.yaml as usual edit the clip: part
2. run the figure
```
MUJOCO_GL=egl .venv/bin/python replay_strobe.py
```
3. freeze the tuning
```
MUJOCO_GL=egl .venv/bin/python replay_strobe.py --save-config configs/strobe/heavy_arm/move-ego-0-2_strobe.yaml
```
4. re-run later (layer: base then your saved overrides, last wins)
```
MUJOCO_GL=egl .venv/bin/python replay_strobe.py --config configs/strobe/heavy_arm/move-ego-0-2_strobe.yaml
```
