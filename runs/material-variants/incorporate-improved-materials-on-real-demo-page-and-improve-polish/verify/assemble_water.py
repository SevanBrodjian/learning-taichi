"""Turn the water rework's captured frames into the media the task page shows.

Every comparison here is BOTH SIDES AGAINST EACH OTHER in the same medium as the claim
(CLAUDE.md -> "Presenting results to the user"), and every one of them holds ONE variable:

  cmp_water.mp4       the demo's own opening scene. Both halves are the SAME build, the SAME
                      physics, the SAME particle positions and the SAME snow/sand/rubber. The only
                      difference is whether the water shades off the iso-surface reconstruction or
                      off four local taps of the raw splat accumulation.
  cmp_water_pool.mp4  the same one-variable comparison on water alone plus one rubber ball, so the
                      surface and the spray are both in frame and nothing else competes for
                      attention.
  still_water_*.png   one frame of each, off the identical particle state.
  cmp_water_target.png  the three-way that answers "did it actually land where the proposal was":
                      T-020's Taichi film render, this build's water, and the water that shipped.
                      Different scenes, and the panel labels say so.

    .venv/Scripts/python.exe runs/.../verify/assemble_water.py
"""
import json
import pathlib
import shutil
import subprocess
import sys

RUN = pathlib.Path(__file__).resolve().parents[1]
OUT = RUN / "verify" / "out"
CAP = OUT / "water"
T020 = RUN.parent / "propose-new-rendering-for-each-of-the-four-materials"
FPS = 24


def ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def pair(tag_a, tag_b, dst, label_a, label_b):
    a, b = CAP / tag_a, CAP / tag_b
    if not (a.exists() and b.exists()):
        print("missing halves for", dst)
        return None
    fs = ("drawtext=text='%s':x=12:y=12:fontsize=20:fontcolor=0xdfe6ee:"
          "box=1:boxcolor=0x0a0e14@0.78:boxborderw=7")
    filt = ("[0:v]" + (fs % label_a) + "[l];[1:v]" + (fs % label_b) + "[r];"
            "[l][r]hstack=inputs=2,scale=trunc(iw/2)*2:trunc(ih/2)*2")
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error",
                    "-framerate", str(FPS), "-i", str(a / "f_%04d.png"),
                    "-framerate", str(FPS), "-i", str(b / "f_%04d.png"),
                    "-filter_complex", filt,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "20",
                    str(dst)], check=True)
    print("wrote", dst)
    return dst


def still(src, dst):
    if not src.exists():
        print("missing still", src)
        return None
    shutil.copyfile(src, dst)
    print("wrote", dst)
    return dst


def contact(dst):
    """T-020's proposal, this build, and the water that shipped -- side by side, labelled.

    Three DIFFERENT scenes: the left panel is T-020's own dam-break at 720 in Taichi and the other
    two are the demo's pool at 600 in WebGPU. That is stated on the panels, because a three-up that
    quietly implies one scene is a misleading figure even when every panel is real.
    """
    from PIL import Image, ImageDraw, ImageFont
    panels = [
        (T020 / "still_fluid_alt.png", "T-020 PROPOSAL  option B film", "Taichi, its own dam-break scene, 720px"),
        (CAP / "still_pool_after.png", "NOW ON THE PAGE  reconstruction", "WebGPU, the demo's pool, 600px"),
        (CAP / "still_pool_before.png", "WHAT T-027 SHIPPED  four taps", "WebGPU, the demo's pool, 600px"),
    ]
    for p, _, _ in panels:
        if not p.exists():
            print("missing panel", p)
            return None
    W, BAND = 460, 52
    ims = [Image.open(p).convert("RGB").resize((W, W), Image.LANCZOS) for p, _, _ in panels]
    out = Image.new("RGB", (W * 3 + 16, W + BAND), (10, 14, 20))
    d = ImageDraw.Draw(out)
    try:
        f1 = ImageFont.truetype("seguisb.ttf", 17)
        f2 = ImageFont.truetype("segoeui.ttf", 14)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    for i, (im, (_, t1, t2)) in enumerate(zip(ims, panels)):
        x = i * (W + 8)
        out.paste(im, (x, BAND))
        d.text((x + 6, 6), t1, font=f1, fill=(223, 230, 238))
        d.text((x + 6, 28), t2, font=f2, fill=(127, 142, 163))
    out.save(dst)
    print("wrote", dst)
    return dst


def main():
    made = []
    made.append(pair("shipped_before", "shipped_after", RUN / "cmp_water.mp4",
                     "BEFORE  water off 4 local taps", "AFTER  screen-space iso-surface"))
    made.append(pair("pool_before", "pool_after", RUN / "cmp_water_pool.mp4",
                     "BEFORE  water off 4 local taps", "AFTER  screen-space iso-surface"))
    made.append(still(CAP / "still_shipped_after.png", RUN / "still_water_after.png"))
    made.append(still(CAP / "still_shipped_before.png", RUN / "still_water_before.png"))
    made.append(contact(RUN / "cmp_water_target.png"))

    summ = {}
    for j in sorted(OUT.glob("water_*.json")):
        summ[j.stem] = json.loads(j.read_text())
    (RUN / "verify" / "water_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\nmade:", [m.name for m in made if m])
    return 0 if all(made) else 1


if __name__ == "__main__":
    sys.exit(main())
