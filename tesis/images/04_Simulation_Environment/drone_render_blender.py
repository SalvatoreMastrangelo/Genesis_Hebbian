"""Step 2 of the drone figures: Cycles render of the geometry dumped by
drone_dump_geometry.py. Run by make_drone_figures.sh with Blender >= 4.2:

    blender -b --python drone_render_blender.py -- GEOMETRY.npz PASS_DIR [--pose neutral] [--ghost] [--top] [--draft]

Writes the RGBA passes that drone_compose.py puts together: hero.png (the drone
in the chosen pose), shadow.png (its shadow on a plane under it) and, with
--ghost, ghost.png (the moving surfaces in their neutral position).
"""
import argparse
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("geometry")
ap.add_argument("out")
ap.add_argument("--pose", choices=("neutral", "actuated"), default="actuated")
ap.add_argument("--ghost", action="store_true")       # extra pass: the moving surfaces in their neutral position
ap.add_argument("--top", action="store_true")         # top view with the nose up, instead of --az and --el
ap.add_argument("--az", type=float, default=148.0)    # yaw of the camera around +Z (nose is +X): from behind, left
ap.add_argument("--el", type=float, default=36.0)     # elevation above the horizontal plane
ap.add_argument("--lens", type=float, default=110.0)  # mm on a 36 mm sensor: long lens, little distortion
ap.add_argument("--ground", type=float, default=0.04)  # shadow plane this far below the drone [m]
ap.add_argument("--draft", action="store_true")       # small and noisy, a few seconds per pass
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args(argv)

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
RES = (1300, 850) if args.draft else (3200, 2100)
SAMPLES = 48 if args.draft else 384
MOVING = ("left_wing", "right_wing", "elevator_hinge", "rudder")   # links drawn in the neutral pose too

# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.use_adaptive_sampling = True
scene.cycles.max_bounces = 6
scene.render.resolution_x, scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "16"
scene.view_settings.view_transform = "Standard"      # keeps the red saturated (AgX and Filmic wash it out)
scene.view_settings.look = "None"

if not args.cpu:      # the first OptiX render on a new GPU compiles kernels for some minutes
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except Exception as exc:
            print("no", kind, exc)
            continue
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            print("cycles device:", kind)
            break

world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.30
scene.world = world


def material(name, rgb, rough, coat, spec):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = 0.08
    return m


# red, grey and black as in the URDF; the light levels below are set for these
# values with the Standard view transform (brighter and the wings clip to white)
MAT = {
    "red": material("red", (0.58, 0.022, 0.022), rough=0.52, coat=0.06, spec=0.25),
    "grey": material("grey", (0.47, 0.49, 0.52), rough=0.52, coat=0.06, spec=0.30),
    "black": material("black", (0.025, 0.027, 0.03), rough=0.30, coat=0.30, spec=0.50),
    "ghost": material("ghost", (0.70, 0.72, 0.75), rough=0.90, coat=0.0, spec=0.10),
}


def mat_of(col):
    r, g, _ = col[:3]
    if r > 0.5 and g < 0.2:
        return MAT["red"]
    return MAT["black"] if r < 0.4 else MAT["grey"]


def shade_smooth(me):
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.set_sharp_from_angle(angle=math.radians(38.0))


def add_mesh(name, v, f, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)       # STL triangles share no vertices
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)             # mirrored parts come inside out
    bm.to_mesh(me)
    bm.free()
    shade_smooth(me)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    return ob


def add_disc(name, v, segments=192):
    """The propeller disc of the URDF is a 32 sided prism: same disc, round rim."""
    lo_d, hi_d = v.min(0), v.max(0)
    axis = int(np.argmin(hi_d - lo_d))
    radius = float(np.delete(hi_d - lo_d, axis).max()) / 2
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius, depth=float((hi_d - lo_d)[axis]),
                                        location=tuple((lo_d + hi_d) / 2))
    ob = bpy.context.active_object
    ob.name = name
    ob.rotation_euler = [(0, math.pi / 2, 0), (math.pi / 2, 0, 0), (0, 0, 0)][axis]
    shade_smooth(ob.data)
    ob.data.materials.append(MAT["black"])
    return ob


data = np.load(args.geometry)
hero, ghost, allv = [], [], []
for key in data.files:
    if not key.endswith("|v"):
        continue
    tag, i, link, _ = key.split("|")
    v, f, col = data[key], data[key[:-1] + "f"], data[key[:-1] + "c"]
    if tag == args.pose:
        if mat_of(col) is MAT["black"]:
            hero.append(add_disc(f"hero_{i}_propeller", v))
        else:
            hero.append(add_mesh(f"hero_{i}_{link}", v, f, mat_of(col)))
        allv.append(v)
    elif args.ghost and tag == "neutral" and link in MOVING:
        ghost.append(add_mesh(f"ghost_{i}_{link}", v, f, MAT["ghost"]))
        allv.append(v)
allv = np.concatenate(allv)
lo, hi = allv.min(0), allv.max(0)
c = (lo + hi) / 2

bpy.ops.mesh.primitive_plane_add(size=30.0, location=(c[0], c[1], lo[2] - args.ground))
ground = bpy.context.active_object
ground.is_shadow_catcher = True

# --------------------------------------------------------------------------- #
# camera: framed on the two poses and on the ground under them
# --------------------------------------------------------------------------- #
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = args.lens
cam_data.clip_start, cam_data.clip_end = 0.05, 100.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
if args.top:
    view, right, up = Vector((0, 0, 1)), Vector((0, -1, 0)), Vector((1, 0, 0))     # nose (+X) up in the image
    cam.rotation_euler = Matrix((right, up, view)).transposed().to_euler()
else:
    az, el = math.radians(args.az), math.radians(args.el)
    view = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    right, up = view.cross(Vector((0, 0, 1))).normalized(), Vector((0, 0, 1))
    cam.rotation_euler = (-view).to_track_quat("-Z", "Y").to_euler()
cam.location = Vector(c) + 6.0 * view
bpy.context.view_layer.update()
pts = [Vector(p) for p in allv[:: max(1, len(allv) // 20000)]]
pts += [Vector((p[0], p[1], lo[2] - args.ground)) for p in allv[:: max(1, len(allv) // 4000)]]
loc, _ = cam.camera_fit_coords(bpy.context.evaluated_depsgraph_get(), [x for p in pts for x in p])
cam.location = Vector(loc) + 0.35 * view      # a little air around the subject
bpy.context.view_layer.update()


# --------------------------------------------------------------------------- #
# lights, placed relative to the camera so that the look survives a change of view
# --------------------------------------------------------------------------- #
def area(name, direction, dist, size, power, colour=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "DISK"
    ld.size = size
    ld.energy = power
    ld.color = colour
    ob = bpy.data.objects.new(name, ld)
    d = Vector(direction).normalized()
    ob.location = Vector(c) + dist * d
    ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(ob)
    return ob


# right = camera right in the world; up = world up, or the nose in the top view
if args.top:
    lights = [area("key", 1.0 * view - 0.45 * right + 0.55 * up, 4.0, 3.0, 80.0, (1.0, 0.98, 0.95)),
              area("fill", 0.8 * view + 1.0 * right - 0.2 * up, 4.5, 4.0, 28.0, (0.95, 0.97, 1.0)),
              area("rim", 0.35 * view + 0.2 * right - 1.0 * up, 4.0, 2.5, 55.0)]
    sun_from = view - 0.22 * right + 0.22 * up          # drop shadow towards the bottom right
else:
    lights = [area("key", 0.55 * view - 0.75 * right + 1.25 * up, 4.0, 3.0, 80.0, (1.0, 0.98, 0.95)),
              area("fill", 0.8 * view + 1.0 * right + 0.35 * up, 4.5, 4.0, 28.0, (0.95, 0.97, 1.0)),
              area("rim", -1.0 * view + 0.2 * right + 0.9 * up, 4.0, 2.5, 55.0)]
    sun_from = 0.45 * view - 0.10 * right + up
# the shadow pass has its own light: a wide sun almost overhead, leaning towards the
# camera, so that the shadow is the planform of the drone and stays tucked under it
sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = 4.0
sun_data.angle = math.radians(30.0)
sun = bpy.data.objects.new("sun", sun_data)
sun.rotation_euler = (-sun_from).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(sun)


def render(name, show, shadow_only=False):
    for ob in hero + ghost:
        ob.hide_render = ob not in show
        ob.visible_camera = not shadow_only
    ground.hide_render = not shadow_only
    sun.hide_render = not shadow_only
    for light in lights:
        light.hide_render = shadow_only
    scene.render.filepath = str(OUT / name)
    bpy.ops.render.render(write_still=True)


render("hero.png", hero)
if ghost:
    render("ghost.png", ghost)
scene.cycles.samples = max(32, SAMPLES // 3)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
render("shadow.png", hero, shadow_only=True)
print("done", OUT)
