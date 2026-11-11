"""skeleton.py -- a body's rest pose as JSON (assets/robots/<body>/skeleton.json), moved from scripts/export_skeleton_json.py (OOP step 9).

dump_skeleton(xml, out) writes the rest-pose (qpos = 0) world transform of every body and geom of an MJCF, computed by MuJoCo
itself (mj_forward), as {"bodies": [...], "geoms": [...]}: name, parent, world_pos, world_quat_wxyz, mass (and the geoms' type,
size, local / world pose). The retargeter reads the root's rest height from it (load_root_rest_height); Blender scenes read the
rest pose. Written exactly as export_skeleton_json.py did (json.dump indent 2, no trailing newline).
"""
import json

import mujoco
import numpy as np


def body_parent_name(model, body_id):
    parent_id = model.body_parentid[body_id]
    if parent_id == body_id:  # worldbody is its own parent
        return None
    return model.body(parent_id).name


def _mat_to_quat(xmat_flat):
    mat = np.array(xmat_flat).reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat.tolist()


def skeleton_dict(model):
    """the skeleton.json content of a compiled model (rest pose: qpos = 0)"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # qpos defaults to 0 -> rest pose

    bodies = []
    for body_id in range(model.nbody):
        name = model.body(body_id).name
        if name == "world":
            continue
        parent = body_parent_name(model, body_id)
        bodies.append({
            "name": name,
            "parent": parent,
            "world_pos": data.xpos[body_id].tolist(),
            "world_quat_wxyz": data.xquat[body_id].tolist(),
            "mass": float(model.body_mass[body_id]),
        })

    geoms = []
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body(body_id).name
        if body_name == "world":
            continue
        geom_type = mujoco.mjtGeom(model.geom_type[geom_id]).name
        geoms.append({
            "name": model.geom(geom_id).name,
            "body": body_name,
            "type": geom_type,
            "size": model.geom_size[geom_id].tolist(),
            "local_pos": model.geom_pos[geom_id].tolist(),
            "local_quat_wxyz": model.geom_quat[geom_id].tolist(),
            "world_pos": data.geom_xpos[geom_id].tolist(),
            "world_quat_wxyz": _mat_to_quat(data.geom_xmat[geom_id]),
        })
    return {"bodies": bodies, "geoms": geoms}


def dump_skeleton(input_path, output_path):
    """export_skeleton_json.py dump_skeleton: MJCF `input_path` -> skeleton.json `output_path`"""
    model = mujoco.MjModel.from_xml_path(str(input_path))
    sk = skeleton_dict(model)
    with open(output_path, "w") as f:
        json.dump(sk, f, indent=2)
    print(f"Wrote {len(sk['bodies'])} bodies, {len(sk['geoms'])} geoms -> {output_path}")
    return output_path


def load_root_rest_height(skeleton_json_path):
    """the root body's rest-pose world z from a skeleton.json (scripts/qpos_retarget.py's)"""
    with open(skeleton_json_path) as f:
        data = json.load(f)
    bodies = {b["name"]: b for b in data["bodies"]}
    root_name = next(b["name"] for b in data["bodies"] if b["parent"] == "world")
    return bodies[root_name]["world_pos"][2]
