"""R2a: VRM batch renderer for 3D->sketch synthesis verification.

Runs inside Blender:
  blender -b -P blender_render.py -- --vrm assets/vrm/fem_vroid.vrm \
      --out r2a/renders --poses stand,sit --builds normal,chibi --cams full,bust

Outputs per scene: line.png (Freestyle contour, white bg), toon.png (shaded),
gt.json (COCO17 pixel keypoints + v flags 0/1/2).
VRM is imported via the native glTF importer (VRM = glb); VRoid J_Bip bone
naming is assumed. Facing / left-right axes are auto-detected at runtime.
"""
import bpy
import json
import math
import os
import random
import shutil
import sys
import tempfile
from math import radians
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

# ---------------- args ----------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


VRM = arg("--vrm", "assets/vrm/fem_vroid.vrm")
OUT = arg("--out", "r2a/renders")
POSES = arg("--poses", "stand,armup,sit,walk").split(",")
BUILDS = arg("--builds", "normal,chibi").split(",")
CAMS = arg("--cams", "full,bust,high").split(",")
SEED = int(arg("--seed", "42"))
RES = (768, 1024)

random.seed(SEED)
MODEL = os.path.splitext(os.path.basename(VRM))[0]
FACE_KP = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}

# ---------------- import ----------------
bpy.ops.wm.read_factory_settings(use_empty=True)

tmp = tempfile.mkdtemp()
glb = os.path.join(tmp, MODEL + ".glb")
shutil.copy(VRM, glb)
bpy.ops.import_scene.gltf(filepath=glb)

ARM = next(o for o in bpy.data.objects if o.type == "ARMATURE")
MESHES = [o for o in bpy.data.objects if o.type == "MESH"]

# bone name candidates: VRoid (J_Bip) first, then Rigify-style
CAND = {
    "hips": ["J_Bip_C_Hips", "hips"], "spine": ["J_Bip_C_Spine", "spine"],
    "chest": ["J_Bip_C_Chest", "chest"], "neck": ["J_Bip_C_Neck", "neck"],
    "head": ["J_Bip_C_Head", "head", "head 1"],
    "eye_l": ["J_Adj_L_FaceEye", "eye.L"], "eye_r": ["J_Adj_R_FaceEye", "eye.R"],
    "uarm_l": ["J_Bip_L_UpperArm", "upper_arm.L"], "uarm_r": ["J_Bip_R_UpperArm", "upper_arm.R"],
    "larm_l": ["J_Bip_L_LowerArm", "forearm.L"], "larm_r": ["J_Bip_R_LowerArm", "forearm.R"],
    "hand_l": ["J_Bip_L_Hand", "hand.L"], "hand_r": ["J_Bip_R_Hand", "hand.R"],
    "uleg_l": ["J_Bip_L_UpperLeg", "thigh.L"], "uleg_r": ["J_Bip_R_UpperLeg", "thigh.R"],
    "lleg_l": ["J_Bip_L_LowerLeg", "shin.L"], "lleg_r": ["J_Bip_R_LowerLeg", "shin.R"],
    "foot_l": ["J_Bip_L_Foot", "foot.L"], "foot_r": ["J_Bip_R_Foot", "foot.R"],
}
B = {}
for key, cands in CAND.items():
    B[key] = next((c for c in cands if c in ARM.pose.bones), None)
required = [k for k in CAND if k not in ("eye_l", "eye_r")]
missing = [k for k in required if B[k] is None]
if missing:
    print("FATAL missing bones:", missing)
    sys.exit(1)
HAS_EYES = B["eye_l"] is not None and B["eye_r"] is not None


def bone_head_world(key):
    pb = ARM.pose.bones[B[key]]
    return ARM.matrix_world @ pb.head


def upd():
    bpy.context.view_layer.update()


# axis auto-detect (rest pose)
upd()
LEFT = 1.0 if bone_head_world("hand_l").x > bone_head_world("hand_r").x else -1.0
if HAS_EYES:
    eye_mid0 = (bone_head_world("eye_l") + bone_head_world("eye_r")) / 2
    FWD = 1.0 if (eye_mid0.y - bone_head_world("head").y) > 0 else -1.0
else:  # feet point forward
    pbf = ARM.pose.bones[B["foot_l"]]
    fdir0 = (ARM.matrix_world @ pbf.tail) - (ARM.matrix_world @ pbf.head)
    FWD = 1.0 if fdir0.y > 0 else -1.0
CHAR_H = bone_head_world("head").z - min(bone_head_world("foot_l").z, bone_head_world("foot_r").z)
HEAD_LEN = (ARM.matrix_world @ ARM.pose.bones[B["head"]].tail
            - ARM.matrix_world @ ARM.pose.bones[B["head"]].head).length
print(f"axes: LEFT(+x?)={LEFT} FWD(+y?)={FWD} height={CHAR_H:.2f} eyes={HAS_EYES}")


def rot_world(key, axis, deg):
    """Rotate pose bone around a world axis at its head."""
    pb = ARM.pose.bones[B[key]]
    upd()
    M = ARM.matrix_world @ pb.matrix
    head = M.to_translation()
    R = Matrix.Translation(head) @ Matrix.Rotation(radians(deg), 4, axis) @ Matrix.Translation(-head)
    pb.matrix = ARM.matrix_world.inverted() @ (R @ M)
    upd()


def aim_bone(key, tdir, j=0.05):
    """Rotate pose bone so its head->tail direction points along world tdir.
    Rest-pose agnostic (works for both T-pose and A-pose rigs)."""
    pb = ARM.pose.bones[B[key]]
    upd()
    M = ARM.matrix_world @ pb.matrix
    head = M.to_translation()
    tail = ARM.matrix_world @ pb.tail
    cur = (tail - head).normalized()
    t = Vector(tdir) + Vector((random.uniform(-j, j), random.uniform(-j, j), random.uniform(-j, j)))
    t.normalize()
    axis = cur.cross(t)
    if axis.length < 1e-6:
        return
    R = (Matrix.Translation(head) @ Matrix.Rotation(cur.angle(t), 4, axis.normalized())
         @ Matrix.Translation(-head))
    pb.matrix = ARM.matrix_world.inverted() @ (R @ M)
    upd()


def reset_pose():
    for pb in ARM.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    upd()


def jitter(d=6):
    return random.uniform(-d, d)


def apply_pose(name):
    """Poses as aim targets (world dirs) + a few world-axis rotations."""
    L, Fy = LEFT, FWD
    F = -FWD  # pitch sign for rot ops
    arms_down = [("aim", "uarm_l", (0.20 * L, 0, -1)), ("aim", "larm_l", (0.16 * L, 0, -1)),
                 ("aim", "uarm_r", (-0.20 * L, 0, -1)), ("aim", "larm_r", (-0.16 * L, 0, -1))]
    p = {
        "stand": arms_down + [("rot", "head", "Z", jitter(10))],
        "armup": [("aim", "uarm_l", (0.30 * L, 0, 1)), ("aim", "larm_l", (0.15 * L, 0, 1)),
                  ("aim", "uarm_r", (-0.20 * L, 0, -1)), ("aim", "larm_r", (-0.16 * L, 0, -1)),
                  ("rot", "head", "Z", jitter(10))],
        "sit": [("aim", "uleg_l", (0.10 * L, Fy, -0.25)), ("aim", "lleg_l", (0.08 * L, 0.05 * Fy, -1)),
                ("aim", "uleg_r", (-0.10 * L, Fy, -0.25)), ("aim", "lleg_r", (-0.08 * L, 0.05 * Fy, -1)),
                ("rot", "spine", "X", -10 * F)] + arms_down,
        "walk": [("aim", "uleg_l", (0.05 * L, 0.55 * Fy, -0.9)), ("aim", "lleg_l", (0.05 * L, 0.1 * Fy, -1)),
                 ("aim", "uleg_r", (-0.05 * L, -0.40 * Fy, -1)), ("aim", "lleg_r", (-0.05 * L, -0.6 * Fy, -0.75)),
                 ("aim", "uarm_l", (0.18 * L, -0.5 * Fy, -1)), ("aim", "larm_l", (0.15 * L, -0.6 * Fy, -1)),
                 ("aim", "uarm_r", (-0.18 * L, 0.5 * Fy, -1)), ("aim", "larm_r", (-0.15 * L, 0.6 * Fy, -1))],
        "wave": [("aim", "uarm_l", (0.20 * L, 0, -1)), ("aim", "larm_l", (0.16 * L, 0, -1)),
                 ("aim", "uarm_r", (-0.75 * L, 0, 0.6)), ("aim", "larm_r", (-0.15 * L, 0, 1)),
                 ("rot", "head", "Z", 12 + jitter(6))],
        "crouch": [("aim", "uleg_l", (0.10 * L, 0.95 * Fy, -0.5)), ("aim", "lleg_l", (0.08 * L, -0.5 * Fy, -1)),
                   ("aim", "uleg_r", (-0.10 * L, 0.95 * Fy, -0.5)), ("aim", "lleg_r", (-0.08 * L, -0.5 * Fy, -1)),
                   ("rot", "spine", "X", -22 * F)] + arms_down,
        "lie": [("rot", "hips", "X", -88 * F)] + arms_down,
    }
    for op in p[name]:
        if op[0] == "aim":
            aim_bone(op[1], op[2])
        else:
            rot_world(op[1], op[2], op[3])


def apply_build(name):
    s = {"normal": {}, "chibi": {"head": 1.9, "uarm_l": 0.62, "uarm_r": 0.62,
                                 "uleg_l": 0.55, "uleg_r": 0.55, "spine": 0.8},
         "small": {"head": 1.4, "uarm_l": 0.8, "uarm_r": 0.8,
                   "uleg_l": 0.75, "uleg_r": 0.75}}[name]
    for key, sc in s.items():
        ARM.pose.bones[B[key]].scale = (sc, sc, sc)
    upd()


# ---------------- keypoints ----------------
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def keypoints_world():
    head = bone_head_world("head")
    if HAS_EYES:
        eye_l, eye_r = bone_head_world("eye_l"), bone_head_world("eye_r")
    else:  # approximate from posed head bone frame
        pb = ARM.pose.bones[B["head"]]
        M = ARM.matrix_world @ pb.matrix
        up = (ARM.matrix_world @ pb.tail - ARM.matrix_world @ pb.head).normalized()
        fw = min((M.to_3x3() @ Vector(ax) for ax in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]),
                 key=lambda v: -(v.to_2d().length))  # most horizontal local axis
        if (fw.x * 0 + fw.y * FWD) < 0:
            fw = -fw
        side = up.cross(fw).normalized()
        eye_mid_a = head + up * HEAD_LEN * 0.55 + fw * HEAD_LEN * 0.45
        eye_l = eye_mid_a + side * HEAD_LEN * 0.16 * LEFT
        eye_r = eye_mid_a - side * HEAD_LEN * 0.16 * LEFT
    eye_mid = (eye_l + eye_r) / 2
    fwd_v = (eye_mid - head)
    nose = eye_mid + fwd_v * 0.35 - Vector((0, 0, 0.30 * (eye_l - eye_r).length))
    side = (eye_l - eye_r).normalized() * (eye_l - eye_r).length * 1.6
    up_off = Vector((0, 0, (eye_l - eye_r).length * 0.6))
    ear_l, ear_r = head + side - fwd_v * 0.1 + up_off, head - side - fwd_v * 0.1 + up_off
    return {
        "nose": nose, "left_eye": eye_l, "right_eye": eye_r,
        "left_ear": ear_l, "right_ear": ear_r,
        "left_shoulder": bone_head_world("uarm_l"), "right_shoulder": bone_head_world("uarm_r"),
        "left_elbow": bone_head_world("larm_l"), "right_elbow": bone_head_world("larm_r"),
        "left_wrist": bone_head_world("hand_l"), "right_wrist": bone_head_world("hand_r"),
        "left_hip": bone_head_world("uleg_l"), "right_hip": bone_head_world("uleg_r"),
        "left_knee": bone_head_world("lleg_l"), "right_knee": bone_head_world("lleg_r"),
        "left_ankle": bone_head_world("foot_l"), "right_ankle": bone_head_world("foot_r"),
    }


def project_and_flag(cam, kw):
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    cam_pos = cam.matrix_world.to_translation()
    out = []
    for name in COCO:
        w = kw[name]
        co = world_to_camera_view(scene, cam, w)
        x, y = co.x * RES[0], (1 - co.y) * RES[1]
        if co.z <= 0 or not (0 <= co.x <= 1 and 0 <= co.y <= 1):
            out.append([round(x, 1), round(y, 1), 0])
            continue
        d = (w - cam_pos)
        dist = d.length
        hit, loc, _, _, _, _ = bpy.context.scene.ray_cast(deps, cam_pos, d.normalized())
        # face bones sit inside the head mesh -> larger tolerance there
        head_scale = ARM.pose.bones[B["head"]].scale[0]
        tol = 0.9 * HEAD_LEN * head_scale if name in FACE_KP else 0.06 * CHAR_H
        v = 2
        if hit and (loc - cam_pos).length < dist - tol:
            v = 1
        out.append([round(x, 1), round(y, 1), v])
    return out


# ---------------- camera / lights ----------------
def make_cam():
    cd = bpy.data.cameras.new("cam")
    cd.lens = 50
    co = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(co)
    bpy.context.scene.camera = co
    return co


def look_at(cam, target):
    d = target - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def place_cam(cam, mode, kw):
    head_scale = ARM.pose.bones[B["head"]].scale[0]
    top = bone_head_world("head") + Vector((0, 0, 1.6 * HEAD_LEN * head_scale))
    pts = list(kw.values()) + [top]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    c = (lo + hi) / 2
    size = max(hi.z - lo.z, (hi - lo).length * 0.8)
    fdir = Vector((random.uniform(-0.25, 0.25), FWD, 0)).normalized()
    if mode == "full":
        cam.location = c + fdir * size * 1.7
        look_at(cam, c)
    elif mode == "bust":
        t = (kw["left_shoulder"] + kw["right_shoulder"]) / 2
        t = t + Vector((0, 0, -0.05 * size))
        cam.location = t + fdir * size * 0.75
        look_at(cam, t)
    elif mode == "high":
        cam.location = c + fdir * size * 1.7 + Vector((0, 0, size * 1.2))
        look_at(cam, c + Vector((0, 0, size * 0.15)))
    upd()


def setup_lights():
    for name, loc, e in [("sun", (2 * FWD, 3 * FWD, 4), 3.0)]:
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = e
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        look_at(lo, Vector((0, 0, 1)))
        bpy.context.collection.objects.link(lo)
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = 1.0
    bpy.context.scene.world = w


# ---------------- render passes ----------------
WHITE_MAT = None


def get_white_mat():
    global WHITE_MAT
    if WHITE_MAT is None:
        m = bpy.data.materials.new("white_emit")
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs[0].default_value = (1, 1, 1, 1)
        outn = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(em.outputs[0], outn.inputs[0])
        WHITE_MAT = m
    return WHITE_MAT


ORIG_MATS = {o.name: [s.material for s in o.material_slots] for o in MESHES}


def set_line_materials(enable):
    for o in MESHES:
        for i, slot in enumerate(o.material_slots):
            slot.material = get_white_mat() if enable else ORIG_MATS[o.name][i]


def pick_engine(names):
    avail = {"CYCLES", "BLENDER_WORKBENCH", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}
    for n in names:
        try:
            bpy.context.scene.render.engine = n
            return n
        except Exception:
            continue
    return bpy.context.scene.render.engine


def render(path, mode):
    sc = bpy.context.scene
    sc.view_settings.view_transform = "Standard"
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.filepath = path
    sc.render.image_settings.file_format = "PNG"
    if mode == "line":
        set_line_materials(True)
        eng = pick_engine(["CYCLES"])
        sc.cycles.samples = 8
        sc.cycles.use_denoising = False
        sc.render.use_freestyle = True
        sc.render.line_thickness = 1.8
        vl = bpy.context.view_layer
        fs = vl.freestyle_settings
        if not fs.linesets:
            ls = fs.linesets.new("ls")
            ls.select_silhouette = ls.select_border = ls.select_crease = True
        fs.crease_angle = radians(120)
        # ensure every lineset in every scene/view layer has a linestyle
        for sc2 in bpy.data.scenes:
            for vl2 in sc2.view_layers:
                vl2.use_freestyle = True
                for ls2 in vl2.freestyle_settings.linesets:
                    if ls2.linestyle is None:
                        ls2.linestyle = bpy.data.linestyles.new("style")
                    ls2.linestyle.color = (0, 0, 0)
                    ls2.linestyle.thickness = 2.2
    else:
        set_line_materials(False)
        eng = pick_engine(["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"])
        sc.render.use_freestyle = False
        if eng == "CYCLES":
            sc.cycles.samples = 24
    bpy.ops.render.render(write_still=True)


# ---------------- main loop ----------------
sc0 = bpy.context.scene
sc0.render.resolution_x, sc0.render.resolution_y = RES  # must be set BEFORE projection
# hide non-humanoid accessory rigs (e.g. seed-san robo arm)
for pb in ARM.pose.bones:
    if pb.name.startswith("robo_"):
        pb.scale = (0.0001, 0.0001, 0.0001)
for o in MESHES:
    if "robo" in o.name.lower():
        o.hide_render = True
upd()
setup_lights()
cam = make_cam()
n = 0
for build in BUILDS:
    for pose in POSES:
        reset_pose()
        apply_build(build)
        apply_pose(pose)
        kw = keypoints_world()
        for cmode in CAMS:
            place_cam(cam, cmode, kw)
            sid = f"{MODEL}_{build}_{pose}_{cmode}"
            d = os.path.join(OUT, sid)
            os.makedirs(d, exist_ok=True)
            kps = project_and_flag(cam, kw)
            with open(os.path.join(d, "gt.json"), "w") as f:
                json.dump({"scene": sid, "model": MODEL, "build": build, "pose": pose,
                           "cam": cmode, "img_w": RES[0], "img_h": RES[1],
                           "coco_order": COCO, "keypoints": kps,
                           "face_kp_approx": True}, f, indent=1)
            render(os.path.join(d, "toon.png"), "toon")
            render(os.path.join(d, "line.png"), "line")
            n += 1
            print("SCENE DONE", sid)
print(f"ALL DONE {n} scenes")
