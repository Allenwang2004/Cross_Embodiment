"""render_blender.py -- render the paper figures' motion panels in Blender (Cycles) from scene.json + meshes.npz in this folder. Runs inside Blender; see README.md.

  blender -b -P render_blender.py -- --panel teaser_g1 --samples 64 --scale 0.5     a quick look at one panel
  blender -b -P render_blender.py --                                                every panel, final quality

Options: --panel NAME (repeatable; default all), --samples N (default 256), --scale X (resolution factor, default 1), --device CPU|GPU (default GPU if Blender finds one), --threads N, --out DIR (default renders).
To change the look, edit the block below."""
import argparse, json, math, os, sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

# ---- the look: change these
BG = (0.737, 0.737, 0.737)        # #bcbcbc, BFMTrack's grey: what poses fade towards
COLORS = dict(body=(0.47, 0.61, 0.70), red=(0.78, 0.15, 0.13), amber=(0.90, 0.60, 0.06), black=(0.09, 0.09, 0.10))
COLORS.update(ref=(0.10, 0.10, 0.10), z0=(0.55, 0.545, 0.53), bfmzero=(0.106, 0.686, 0.478), ours=(0.165, 0.471, 0.839))   # top-down comparisons (scripts/topdown_blender.py), the plots' colours
FLOOR_COLOR = (0.74, 0.74, 0.74); FLOOR_ROUGHNESS = 0.5
FLOOR_LIFT = 0.08                 # a panel marked "white": white floor, poses fade to white, this much glow takes the lit floor past pure white while shadows stay
BODY_ROUGHNESS = 0.42
WORLD_STRENGTH = 0.5
KEY_ENERGY = 1.75; KEY_ANGLE = 14.0; KEY_POS = (-2.2, 2.2, 6.5)      # a soft sun, the same on every panel: KEY_POS sets the direction of the shadow, KEY_ANGLE its softness
CAM_DISTANCE = 14.0; AZIMUTH = 125.0; ELEVATION = -8.0              # same view as the MuJoCo figures

HERE = os.path.dirname(os.path.abspath(__file__))


def srgb_to_linear(c):
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def material(cache, name, rgb, roughness, alpha=1.0):
    key = (name, tuple(np.round(rgb, 4)), roughness, alpha)
    if key not in cache:
        m = bpy.data.materials.new(f"{name}_{len(cache)}"); m.use_nodes = True; p = m.node_tree.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = (*srgb_to_linear(rgb), 1); p.inputs["Roughness"].default_value = roughness; p.inputs["Metallic"].default_value = 0.0
        if alpha < 1:                      # a see-through ghost: only the nearest ghost surface along a ray counts, so it shows as one silhouette, not as its insides
            t = m.node_tree; lp = t.nodes.new("ShaderNodeLightPath"); near = t.nodes.new("ShaderNodeMath"); near.operation = "LESS_THAN"; near.inputs[1].default_value = 0.5; mul = t.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = alpha
            t.links.new(lp.outputs["Transparent Depth"], near.inputs[0]); t.links.new(near.outputs[0], mul.inputs[0]); t.links.new(mul.outputs[0], p.inputs["Alpha"])
        cache[key] = m
    return cache[key]


def lathe(r_of_z, n=48):
    """a closed surface of revolution around z. r_of_z: (z, r) from bottom to top."""
    verts, faces = [], []
    for z, r in r_of_z:
        verts += [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    m = len(r_of_z)
    for k in range(m - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n; faces.append((a, b, b + n, a + n))
    faces += [tuple(range(n))[::-1], tuple((m - 1) * n + i for i in range(n))]
    return verts, faces


def primitive(kind, size):
    """vertices and faces of a MuJoCo geom in its own frame."""
    arc = lambda r, z0, a, b, k=12: [(z0 + r * math.sin(a + (b - a) * i / k), r * math.cos(a + (b - a) * i / k)) for i in range(k + 1)]
    if kind == "box":
        x, y, z = size; v = [(sx * x, sy * y, sz * z) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
        return v, [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)], False
    if kind == "sphere":
        return (*lathe(arc(size[0], 0, -math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, 24)), True)
    if kind == "ellipsoid":
        v, f = lathe(arc(1.0, 0, -math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, 24)); return [(a * size[0], b * size[1], c * size[2]) for a, b, c in v], f, True
    if kind == "capsule":
        r, h = size[0], size[1]; return (*lathe(arc(r, -h, -math.pi / 2 + 1e-3, 0) + arc(r, h, 0, math.pi / 2 - 1e-3)), True)
    if kind == "cylinder":
        r, h = size[0], size[1]; return (*lathe([(-h, r), (h, r)]), True)
    raise ValueError(kind)


def make_mesh(name, verts, faces, smooth):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces]); me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons)); me.update()
    return me


def build_panel(name, p, meshes, samples, scale, out):
    bpy.ops.wm.read_factory_settings(use_empty=True); s = bpy.context.scene; s.render.engine = "CYCLES"; configure_device(s)
    s.cycles.samples = samples; s.cycles.use_denoising = True; s.cycles.transparent_max_bounces = 64; s.render.resolution_x, s.render.resolution_y = int(p["px"][0] * scale), int(p["px"][1] * scale); s.render.resolution_percentage = 100
    cutout = p.get("cutout", False); s.render.film_transparent = cutout; s.render.image_settings.file_format = "PNG"; s.render.image_settings.color_mode = "RGBA" if cutout else "RGB"; s.view_settings.view_transform = "Standard"; s.view_settings.look = "None"; s.render.filepath = os.path.join(out, name + ".png")
    w = bpy.data.worlds.new("world"); s.world = w; w.use_nodes = True; bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = WORLD_STRENGTH
    mats, shapes = {}, {}
    floor = make_mesh("floor", [(-200, -200, 0), (200, -200, 0), (200, 200, 0), (-200, 200, 0)], [(0, 1, 2, 3)], False); o = bpy.data.objects.new("floor", floor); s.collection.objects.link(o); white = p.get("white", False); fm = material(mats, "floor", (1.0, 1.0, 1.0) if white else FLOOR_COLOR, FLOOR_ROUGHNESS); floor.materials.append(fm); o.is_shadow_catcher = cutout; fade = np.array((1.0, 1.0, 1.0) if white else BG)
    if white:
        fp = fm.node_tree.nodes["Principled BSDF"]; fp.inputs["Emission Color"].default_value = (1, 1, 1, 1); fp.inputs["Emission Strength"].default_value = FLOOR_LIFT
    xs = []; el, az = math.radians(p.get("elevation", ELEVATION)), math.radians(AZIMUTH); right = Vector((math.sin(az), -math.cos(az), 0)); D = p.get("cam_distance", CAM_DISTANCE)      # a pose's "shift": metres sideways in the image, for several bodies on one floor
    for pose in p["poses"]:
        tint = pose["tint"]; xs.append(pose["x"]); across = right * pose.get("shift", 0.0)
        for g in pose["geoms"]:
            rgb = tuple((1 - tint) * np.array(COLORS[g["color"]]) + tint * fade); mat = material(mats, g["color"], rgb, BODY_ROUGHNESS, pose.get("alpha", 1.0))
            key = (g["kind"], tuple(g["size"]), g.get("mesh"), mat.name)
            if key not in shapes:
                if g["kind"] == "mesh":
                    shapes[key] = make_mesh(g["mesh"], meshes[g["mesh"] + "__v"], meshes[g["mesh"] + "__f"], True)
                else:
                    v, f, sm = primitive(g["kind"], g["size"]); shapes[key] = make_mesh(g["kind"], v, f, sm)
                shapes[key].materials.append(mat)
            ob = bpy.data.objects.new(g["kind"], shapes[key]); s.collection.objects.link(ob)
            R = g["R"]; ob.matrix_world = Matrix(((R[0], R[1], R[2], g["p"][0] + across.x), (R[3], R[4], R[5], g["p"][1] + across.y), (R[6], R[7], R[8], g["p"][2]), (0, 0, 0, 1)))
            if g["kind"] == "mesh":
                ob.modifiers.new("split", "EDGE_SPLIT").split_angle = math.radians(35)
    # camera: the panel's width in metres is what the lens sees, the floor sits at feet_m above the bottom edge
    W, H = p["width_m"], p["height_m"]; fwd = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))); zc = (H / 2 - p["feet_m"]) / math.cos(el); look = Vector((sum(xs) / len(xs) * 1.0, 0, zc))
    cam = bpy.data.cameras.new("cam"); cam.sensor_fit = "HORIZONTAL"; cam.sensor_width = 36; cam.lens = 36 * D / W; cam.clip_end = 10 * D; co = bpy.data.objects.new("cam", cam); s.collection.objects.link(co); s.camera = co
    co.location = look - D * fwd; co.rotation_mode = "QUATERNION"; co.rotation_quaternion = fwd.to_track_quat("-Z", "Y")
    li = bpy.data.lights.new("key", "SUN"); li.energy = KEY_ENERGY; li.angle = math.radians(KEY_ANGLE); lo = bpy.data.objects.new("key", li); s.collection.objects.link(lo); lo.location = (look.x + KEY_POS[0], KEY_POS[1], KEY_POS[2]); lo.rotation_mode = "QUATERNION"; lo.rotation_quaternion = (Vector((look.x, 0, 0.8)) - lo.location).to_track_quat("-Z", "Y")
    bpy.ops.render.render(write_still=True); print("saved", s.render.filepath, flush=True)


ARGS = None


def configure_device(s):
    a = ARGS
    if a.threads:
        s.render.threads_mode = "FIXED"; s.render.threads = a.threads
    s.cycles.device = "CPU"
    if a.device == "CPU":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "METAL", "ONEAPI", "HIP"):
        try:
            prefs.compute_device_type = kind; prefs.get_devices(); gpus = [d for d in prefs.devices if d.type == kind]
        except (TypeError, AttributeError):
            continue
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            s.cycles.device = "GPU"; print("device", kind, [d.name for d in gpus], flush=True); return
    if a.device == "GPU":
        print("no GPU found, using the CPU", flush=True)


def main():
    global ARGS
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(); ap.add_argument("--panel", action="append"); ap.add_argument("--samples", type=int, default=256); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--device", default="AUTO", choices=["AUTO", "CPU", "GPU"]); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--out", default=os.path.join(HERE, "renders")); ARGS = ap.parse_args(argv)
    scene = json.load(open(os.path.join(HERE, "scene.json"))); meshes = dict(np.load(os.path.join(HERE, "meshes.npz"))); os.makedirs(ARGS.out, exist_ok=True)
    names = ARGS.panel or list(scene["panels"]); print("panels:", names, flush=True)
    for n in names:
        build_panel(n, scene["panels"][n], meshes, ARGS.samples, ARGS.scale, ARGS.out)


if __name__ == "__main__":
    main()
