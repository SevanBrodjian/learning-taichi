"""Turn the captured PNG frame sequences into the before/after clips the task page shows.

Every comparison on the page is BOTH SIDES AGAINST EACH OTHER, in the same medium as the claim
(CLAUDE.md -> "Presenting results to the user"). These are claims about MOTION and about APPEARANCE
over time, so they are videos, stacked side by side with a label burnt in, never two stills.

    .venv/Scripts/python.exe runs/.../verify/assemble.py
"""
import json
import pathlib
import shutil
import subprocess
import sys

RUN = pathlib.Path(__file__).resolve().parents[1]
CAP = RUN / "verify" / "out" / "cap"
FPS = 24


def ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def one(tag, dst):
    src = CAP / tag
    frames = sorted(src.glob("f_*.png"))
    if not frames:
        print("no frames for", tag)
        return None
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(src / "f_%04d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "20",
                    str(dst)], check=True)
    print("wrote", dst, len(frames), "frames")
    return dst


def pair(tag_a, tag_b, dst, label_a, label_b):
    """Side by side, each half labelled. The label is burnt in rather than left to the page, because
    the clip has to stay readable when it is opened on its own out of the run directory."""
    a, b = CAP / tag_a, CAP / tag_b
    if not (a.exists() and b.exists()):
        print("missing halves for", dst)
        return None
    fs = "drawtext=text='%s':x=12:y=12:fontsize=22:fontcolor=0xdfe6ee:box=1:boxcolor=0x0a0e14@0.75:boxborderw=7"
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


def still(tag, frame, dst):
    src = CAP / tag / ("f_%04d.png" % frame)
    if not src.exists():
        print("missing still", src)
        return None
    shutil.copyfile(src, dst)
    print("wrote", dst)
    return dst


def main():
    made = []
    # ONE VARIABLE PER COMPARISON. Both of these could have been made by simply running the old page
    # and the new page side by side, and both would then have changed two things at once -- which is
    # not evidence of either. So each clip holds everything else fixed:
    #
    #   cmp_buoyancy  same initial conditions, same substep schedule, BOTH DRAWN WITH THE OLD
    #                 SHADING. Only the physics differs.
    #   cmp_render    the same build and the same physics on both sides, drawn once with the
    #                 previously shipped treatment and once with this task's. Only the shading
    #                 differs -- the particles are in the same place in both halves.
    #   cmp_page      the honest whole-page before/after, both changes at once, for the reader who
    #                 wants to know what actually changed on the page they open.
    made.append(pair("old_three", "new_three_mvp", RUN / "cmp_buoyancy.mp4",
                     "BEFORE  phys-bebeaafbe73e", "AFTER  phys-c518316a4a05"))
    made.append(pair("new_shipped_mvp", "new_shipped", RUN / "cmp_render.mp4",
                     "BEFORE  one treatment", "AFTER  four treatments"))
    made.append(pair("old_shipped", "new_shipped", RUN / "cmp_page.mp4",
                     "BEFORE  the shipped page", "AFTER  this task"))
    for t, d in (("old_three", "still_buoyancy_before.png"), ("new_three_mvp", "still_buoyancy_after.png")):
        made.append(still(t, 71, RUN / d))
    for t, d in (("new_shipped_mvp", "still_render_before.png"), ("new_shipped", "still_render_after.png")):
        made.append(still(t, 89, RUN / d))
    made.append(one("new_shipped", RUN / "demo_render_after.mp4"))
    made.append(one("new_three", RUN / "demo_buoyancy_after.mp4"))
    summ = {}
    for j in sorted((RUN / "verify" / "out").glob("cap_*.json")):
        summ[j.stem] = json.loads(j.read_text())
    (RUN / "verify" / "capture_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\nmade:", [str(m.name) for m in made if m])
    return 0 if all(made) else 1


if __name__ == "__main__":
    sys.exit(main())
