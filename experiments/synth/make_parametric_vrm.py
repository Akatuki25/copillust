"""[不採用 2026-07-04] Generate parametric mannequin glbs via the Blender VRM add-on.

RETIRED as a condition-asset source: the skin-modifier body deviates too far from
human anatomy (no hands/feet/face, detached shoulders) and degrades diffusion
conditioning — MistoLine/NoobAI expect human-shaped contours. Base bodies come
from VRoid Studio exports instead. Kept for the skeleton-level (§3b) work where
render quality is irrelevant, and for the hard-won gotchas in the comments
(Skin modifier collapses rootless components; icyp leaves a stray Icosphere;
wip_with_template_mesh is dead code).

  blender -b -P make_parametric_vrm.py -- [--out assets/vrm]

Uses bpy.ops.icyp.make_basic_armature (VRM add-on) — head_ratio is 頭身.
The add-on's wip_with_template_mesh option is dead code (declared, never used by
execute), so the body mesh is built here: skin-modifier tube mannequin over the
bone graph — i.e. a デッサン人形, which is the right look for (b)-style
"draw a proper character over the mannequin" diffusion conditioning.
Filenames carry the build suffix (_normal/_chibi) which blender_render.py
treats as a native build (no bone-scale faking).
"""
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


OUT = arg("--out", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "vrm"))

SPECS = {
    "fem2_normal": dict(tall=1.58, head_ratio=6.8, shoulder_width=0.072),
    "masc2_normal": dict(tall=1.76, head_ratio=7.2, shoulder_width=0.095,
                         shoulder_in_width=0.06),
    "fem2_chibi": dict(tall=0.85, head_ratio=2.8, head_width_ratio=0.85),
    "masc2_chibi": dict(tall=0.92, head_ratio=3.0, head_width_ratio=0.85),
}

# base radii in meters at tall=1.6: ((x_head, y_head), (x_tail, y_tail)) per bone —
# x is body width, y is body depth (torso is an ellipse, limbs round). .L/.R shared
SKIN = {
    "hips": ((0.105, 0.070), (0.095, 0.065)),
    "spine": ((0.095, 0.065), (0.090, 0.060)),
    "chest": ((0.090, 0.060), (0.105, 0.065)),
    "neck": ((0.030, 0.030), (0.028, 0.028)),
    "head": (None, None),  # set from head unit
    "shoulder": ((0.045, 0.045), (0.042, 0.042)),
    "upper_arm": ((0.040, 0.040), (0.032, 0.032)),
    "lower_arm": ((0.030, 0.030), (0.022, 0.022)),
    "hand": ((0.020, 0.020), (0.016, 0.016)),
    "upper_leg": ((0.075, 0.075), (0.052, 0.052)),
    "lower_leg": ((0.050, 0.050), (0.028, 0.028)),
    "foot": ((0.028, 0.028), (0.020, 0.020)),
}


def build_mannequin(arm, tall, head_ratio):
    """Skin-modifier tube body over the bone graph + auto weights."""
    head_unit = tall / head_ratio
    chunk = (7.0 / head_ratio) ** 0.5  # chibi limbs proportionally chubbier
    verts, edges, radii = [], [], []
    vidx = {}

    def vert(pos, r):
        key = (round(pos.x, 4), round(pos.y, 4), round(pos.z, 4))
        if key in vidx:
            i = vidx[key]
            radii[i] = (max(radii[i][0], r[0]), max(radii[i][1], r[1]))
            return i
        vidx[key] = len(verts)
        verts.append(pos.copy())
        radii.append(r)
        return vidx[key]

    for bn in arm.data.bones:
        base = bn.name.split(".")[0]
        if base not in SKIN:
            continue
        if base == "head":
            rh = (0.36 * head_unit, 0.36 * head_unit)
            rt = (0.30 * head_unit, 0.30 * head_unit)
        else:
            k = tall / 1.6 * (chunk if "arm" in base or "leg" in base
                              or base in ("hand", "foot") else 1.0)
            rh = tuple(r * k for r in SKIN[base][0])
            rt = tuple(r * k for r in SKIN[base][1])
        edges.append((vert(bn.head_local, rh), vert(bn.tail_local, rt)))

    # bridge limbs to the torso: the Skin modifier collapses connected components
    # that lack a root vertex to zero radius, so the graph must be one component
    bb = {bn.name: bn for bn in arm.data.bones}
    for a, aend, b, bend in [("chest", "tail_local", "shoulder.L", "head_local"),
                             ("chest", "tail_local", "shoulder.R", "head_local"),
                             ("hips", "head_local", "upper_leg.L", "head_local"),
                             ("hips", "head_local", "upper_leg.R", "head_local")]:
        pa, pb = getattr(bb[a], aend), getattr(bb[b], bend)
        edges.append((vert(pa, (0.01, 0.01)), vert(pb, (0.01, 0.01))))
    root_i = vidx[(round(bb["hips"].head_local.x, 4), round(bb["hips"].head_local.y, 4),
                   round(bb["hips"].head_local.z, 4))]

    me = bpy.data.meshes.new("body")
    me.from_pydata([v[:] for v in verts], edges, [])
    ob = bpy.data.objects.new("body", me)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    skin = ob.modifiers.new("skin", "SKIN")
    skin.use_smooth_shade = True
    for i, sv in enumerate(ob.data.skin_vertices[0].data):
        sv.radius = radii[i]
    ob.data.skin_vertices[0].data[root_i].use_root = True
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 3  # 2 leaves visible facets in bust closeups
    bpy.ops.object.modifier_apply(modifier="skin")
    bpy.ops.object.modifier_apply(modifier="sub")
    mat = bpy.data.materials.new("body_mat")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = \
        (0.87, 0.80, 0.74, 1.0)
    ob.data.materials.append(mat)
    # rigid nearest-segment binding (bone heat fails on this mesh and then the
    # VRM exporter silently drops the whole skin — skins: 0 in the glb)
    def seg_dist(p, a, b):
        ab = b - a
        t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-9)))
        return (a + ab * t - p).length

    seg_bones = [bn for bn in arm.data.bones if bn.name.split(".")[0] in SKIN]
    groups = {bn.name: ob.vertex_groups.new(name=bn.name) for bn in seg_bones}
    for v in ob.data.vertices:
        best = min(seg_bones, key=lambda bn: seg_dist(v.co, bn.head_local, bn.tail_local))
        groups[best.name].add([v.index], 1.0, "REPLACE")
    ob.parent = arm
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    return ob


os.makedirs(OUT, exist_ok=True)
for name, kw in SPECS.items():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # factory reset disables extensions — re-enable the VRM add-on each round
    bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.vrm")
    bpy.ops.icyp.make_basic_armature(**kw)
    # the icyp operator leaves a stray 2m Icosphere in the scene — drop it
    for o in list(bpy.data.objects):
        if o.type == "MESH":
            bpy.data.objects.remove(o)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    build_mannequin(arm, kw["tall"], kw["head_ratio"])
    objs = [(o.name, o.type) for o in bpy.data.objects]
    print(name, "objects:", objs)
    # plain glTF, not VRM: the renderer only needs bones+mesh and imports via the
    # glTF importer anyway; the VRM exporter's bone normalization broke the skin
    path = os.path.join(OUT, name + ".glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")
    print("EXPORTED", path, os.path.getsize(path))
print("ALL DONE")
