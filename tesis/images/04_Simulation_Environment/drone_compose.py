"""Step 3 of the drone figures: put the passes of drone_render_blender.py together
on white. From the back: shadow, neutral pose if there is one (pale, with an
outline thick enough to survive print at 8 cm), drone.

    python drone_compose.py PASS_DIR OUT.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument("passes")
ap.add_argument("out")
ap.add_argument("--width", type=int, default=1800)
ap.add_argument("--ghost", type=float, default=0.62)     # opacity of the neutral pose
ap.add_argument("--shadow", type=float, default=0.15)    # strength of the ground shadow
ap.add_argument("--line", type=float, default=0.55)      # strength of the outline of the neutral pose
args = ap.parse_args()
P = Path(args.passes)


def load(name):
    a = np.asarray(Image.open(P / name).convert("RGBA")).astype(np.float32) / 255.0
    return a[..., :3], a[..., 3:4]


hero, a_h = load("hero.png")
ghost, a_g = load("ghost.png") if (P / "ghost.png").exists() else (np.ones_like(hero), np.zeros_like(a_h))
_, a_s = load("shadow.png")
H, W = a_h.shape[:2]

# the shadow never reaches the border of the frame; what the denoiser leaves there is noise
m = int(0.04 * W)
win = np.minimum(np.clip(np.minimum(np.arange(W), W - 1 - np.arange(W)) / m, 0, 1)[None, :],
                 np.clip(np.minimum(np.arange(H), H - 1 - np.arange(H)) / m, 0, 1)[:, None])
a_s = a_s * win[..., None]

out = np.ones_like(hero)
out = out * (1 - args.shadow * a_s) + np.array([0.10, 0.11, 0.14], np.float32) * args.shadow * a_s

fill = 0.35 * ghost + 0.65
out = out * (1 - args.ghost * a_g) + fill * args.ghost * a_g
mask = Image.fromarray((a_g[..., 0] > 0.5).astype(np.uint8) * 255)
w = max(1, round(W * 0.0011))
edge = np.asarray(mask.filter(ImageFilter.MaxFilter(2 * w + 1))).astype(np.float32) - \
    np.asarray(mask.filter(ImageFilter.MinFilter(2 * w + 1))).astype(np.float32)
edge = np.asarray(Image.fromarray(np.clip(edge, 0, 255).astype(np.uint8))
                  .filter(ImageFilter.GaussianBlur(0.6 * w))).astype(np.float32)[..., None] / 255.0
out = out * (1 - args.line * edge) + np.array([0.52, 0.54, 0.58], np.float32) * args.line * edge

out = out * (1 - a_h) + hero * a_h

ink = (a_h[..., 0] > 0.02) | (a_g[..., 0] > 0.02) | (args.shadow * a_s[..., 0] > 0.01)
ys, xs = np.where(ink)
pad = int(0.015 * W)
y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
img = Image.fromarray((np.clip(out[y0:y1, x0:x1], 0, 1) * 255 + 0.5).astype(np.uint8))
if img.width > args.width:
    img = img.resize((args.width, round(img.height * args.width / img.width)), Image.LANCZOS)
img.save(args.out, optimize=True)
print("wrote", args.out, img.size)
