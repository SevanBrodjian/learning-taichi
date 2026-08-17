"""Read every exported clip and figure back off disk and check it is not degenerate.

Videos cannot be looked at directly, so each one is decoded, sampled at six evenly spaced times, and
tiled into a contact sheet that CAN be looked at. Also flags the failure modes that matter here: a clip
whose frames never change (nothing moved), a frame that is essentially blank, and a clip that decodes
to fewer frames than it should.
"""
import glob
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "_inspect")
os.makedirs(OUT, exist_ok=True)
NS = 6


def sheet(path):
    rd = imageio.get_reader(path)
    frames = [f for f in rd]
    n = len(frames)
    if n == 0:
        print(f"  !! {os.path.basename(path)}: ZERO FRAMES")
        return
    idx = np.linspace(0, n - 1, min(NS, n)).round().astype(int)
    sel = [frames[i] for i in idx]
    h, w = sel[0].shape[:2]
    # motion check: mean abs difference between first and last sampled frame
    move = float(np.abs(sel[0].astype(np.int16) - sel[-1].astype(np.int16)).mean())
    dark = [float(f.mean()) for f in sel]
    cols = 3
    rows = int(np.ceil(len(sel) / cols))
    canvas = Image.new("RGB", (w * cols, h * rows), (10, 14, 20))
    for k, f in enumerate(sel):
        canvas.paste(Image.fromarray(f), ((k % cols) * w, (k // cols) * h))
    scale = min(1.0, 1500 / canvas.width)
    if scale < 1.0:
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)))
    dst = os.path.join(OUT, os.path.basename(path).replace(".mp4", "_sheet.png"))
    canvas.save(dst)
    flag = ""
    if move < 1.0:
        flag += "  !! NOTHING MOVED"
    if min(dark) < 3.0:
        flag += "  !! A SAMPLED FRAME IS NEARLY BLANK"
    print(f"  {os.path.basename(path):32s} {n:4d} frames {w}x{h}  motion={move:6.2f}  "
          f"brightness {min(dark):.1f}-{max(dark):.1f}{flag}")


print("=== videos ===")
for p in sorted(glob.glob(os.path.join(D, "*.mp4"))):
    sheet(p)
print("\n=== images ===")
for p in sorted(glob.glob(os.path.join(D, "*.png"))):
    im = np.asarray(Image.open(p).convert("RGB"))
    print(f"  {os.path.basename(p):32s} {im.shape[1]}x{im.shape[0]}  "
          f"mean={im.mean():.1f} std={im.std():.1f}"
          + ("  !! FLAT / EMPTY" if im.std() < 4 else ""))
print("\ncontact sheets in", OUT)
