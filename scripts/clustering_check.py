# Cluster the exported latent z vectors and visualize them with PCA.
import re
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
data_root = ROOT / "data"
sources = {
    "z": data_root / "origin_z",
    "infer_origin_z": data_root / "infer_origin_z",
    "infer_retargeting_z": data_root / "infer_retargeting_z",
}


def natural_key(path: Path):
    """Sort trailing digits numerically so _10 does not come before _9."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.stem)]


def prefix_of(folder_name: str) -> str:
    """Use the first segment of a folder name as its group (move-ego-0-2 -> move)."""
    return folder_name.split("-")[0]


# ==========================================
# 1. Pick folders: keep only one folder per shared prefix (e.g. all move-*)
# ==========================================
group_to_folder = {}
for src_dir in sources.values():
    if not src_dir.exists():
        continue
    for folder in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        group_to_folder.setdefault(prefix_of(folder.name), folder.name)

selected_folder = [group_to_folder[g] for g in sorted(group_to_folder)]
print(f"Selected folders ({len(selected_folder)}): {selected_folder}")

# ==========================================
# 2. Load data: one mean z per .npy, averaged over the whole trajectory
# ==========================================
all_vectors = []
folder_labels = []   # which folder the vector belongs to (color)
source_labels = []   # which source the vector came from (one figure per source)

for src_idx, (src_name, src_dir) in enumerate(sources.items()):
    if not src_dir.exists():
        print(f"Warning: Source {src_dir} does not exist, skipping...")
        continue

    for folder_idx, folder_name in enumerate(selected_folder):
        target_path = src_dir / folder_name

        if not target_path.exists():
            print(f"Warning: Folder {target_path} does not exist, skipping...")
            continue

        npy_files = sorted(target_path.glob("*.npy"), key=natural_key)
        if not npy_files:
            print(f"Warning: No .npy under {target_path}, skipping...")
            continue

        # Each file is one individual retargeting run; take the mean z over the
        # whole trajectory. The last frame is the goal embedding (tracking_inference's
        # sliding window is down to a single frame), which is semantically different
        # from the reward embedding in data/z. After averaging we project back onto
        # the sphere of radius sqrt(d) so it lives in the same space as
        # project_z / data/z.
        for npy_file in npy_files:
            vec = np.load(npy_file)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)

            z = vec.mean(axis=0)
            z = z / np.linalg.norm(z) * np.sqrt(z.shape[0])
            all_vectors.append(z)
            folder_labels.append(folder_idx)
            source_labels.append(src_idx)

if not all_vectors:
    raise ValueError("No .npy files loaded. Please check your data_dir path and folder names.")

X = np.vstack(all_vectors)
folder_labels = np.asarray(folder_labels)
source_labels = np.asarray(source_labels)

print(f"Total loaded samples: {X.shape[0]}, Feature dimension: {X.shape[1]}")

# ==========================================
# 3. Reduce to 2D with PCA (fit on all data so the three figures share one projection)
# ==========================================
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

# ==========================================
# 4. Visualization: one figure per source
# ==========================================
out_dir = ROOT / "outputs" / Path(__file__).stem
out_dir.mkdir(parents=True, exist_ok=True)

colors = plt.cm.tab10(np.linspace(0, 1, len(selected_folder)))
source_names = list(sources)

# Shared axis limits so the three figures can be compared side by side.
pad = 0.05 * np.ptp(X_2d, axis=0)
xlim = (X_2d[:, 0].min() - pad[0], X_2d[:, 0].max() + pad[0])
ylim = (X_2d[:, 1].min() - pad[1], X_2d[:, 1].max() + pad[1])


def plot(src_idx, src_name):
    plt.figure(figsize=(10, 8))

    for folder_idx, folder_name in enumerate(selected_folder):
        mask = (folder_labels == folder_idx) & (source_labels == src_idx)
        if not mask.any():
            continue
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[colors[folder_idx]],
            marker="o",
            label=folder_name,
            alpha=0.8,
            edgecolors="none",
            s=80,
        )

    plt.title(f"2D PCA of Mean-z Vectors - {src_name}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = out_dir / f"{src_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


for src_idx, src_name in enumerate(source_names):
    plot(src_idx, src_name)
