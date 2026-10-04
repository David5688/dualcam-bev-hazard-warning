"""Stitch the demo clips into one presentation reel with title cards.

Narrative:  main title -> 3D reconstruction -> system overview -> hazard warning
            -> uncertainty-aware warning -> closing card.

All inputs differ in resolution, so each is normalised to 1280x720 / 30fps (scaled,
letterboxed on black) and re-encoded to identical params, then concatenated. Title
cards are drawn with OpenCV (ASCII only -- localise to Chinese in the slides).

  python stitch_presentation.py   # -> results\PRESENTATION_reel.mp4
"""
import os, sys, cv2, subprocess, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "E:/miniforge3/envs/lss/lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
W, HH = 1280, 720
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
TMP = os.path.join(RESULTS, "_stitch")
os.makedirs(TMP, exist_ok=True)


def title_card(path, title, subtitle="", tag="", secs=2.6):
    img = np.full((HH, W, 3), 20, np.uint8)
    cv2.line(img, (W // 2 - 300, 300), (W // 2 + 300, 300), (90, 90, 90), 1)
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
    cv2.putText(img, title, ((W - tw) // 2, 270), cv2.FONT_HERSHEY_DUPLEX, 1.6, (245, 245, 245), 3, cv2.LINE_AA)
    if subtitle:
        (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(img, subtitle, ((W - sw) // 2, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2, cv2.LINE_AA)
    if tag:
        (gw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(img, tag, ((W - gw) // 2, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 180, 240), 1, cv2.LINE_AA)
    png = path + ".png"
    cv2.imwrite(png, img)
    subprocess.run([FFMPEG, "-y", "-loop", "1", "-t", str(secs), "-i", png, "-r", "30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"scale={W}:{HH}", path],
                   check=True, capture_output=True)
    os.remove(png)


def normalise(src, dst):
    vf = (f"scale={W}:{HH}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{HH}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=30")
    subprocess.run([FFMPEG, "-y", "-i", src, "-vf", vf, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", dst], check=True, capture_output=True)


def main():
    r = RESULTS
    segments = [
        ("00_title", None, ("Dual-Camera 3D Reconstruction", "& Driving Hazard Warning",
                            "front + rear cameras  |  CARLA")),
        ("01_t", None, ("1.  3D Reconstruction", "front + rear GT depth, accumulated over the drive", "")),
        ("01_v", os.path.join(r, "RECON_spin_crisp.mp4"), None),
        ("02_t", None, ("2.  System Overview", "cameras + point cloud + BEV hazard radar", "")),
        ("02_v", os.path.join(r, "INTEGRATED_demo_town01.mp4"), None),
        ("03_t", None, ("3.  Hazard Warning", "clean BEV + TTC-based front/rear warnings", "")),
        ("03_v", os.path.join(r, "HAZARD_BEV_scene.mp4"), None),
        ("04_t", None, ("4.  Uncertainty-Aware Warning", "warn only when confident  (novelty)", "")),
        ("04_v", os.path.join(r, "UNCERT_BEV_town01.mp4"), None),
        ("05_end", None, ("Trustworthy Warning", "variant-D (blind zone) + uncertainty gating",
                          "false alarms 31 -> 5")),
    ]
    parts = []
    for name, src, card in segments:
        dst = os.path.join(TMP, name + ".mp4")
        if src is None:
            title_card(dst, card[0], card[1], card[2])
        else:
            if not os.path.exists(src):
                print("MISSING, skipped:", src); continue
            normalise(src, dst)
        parts.append(dst)
        print("prepared:", name)

    listf = os.path.join(TMP, "concat.txt")
    with open(listf, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    out = os.path.join(r, "PRESENTATION_reel.mp4")
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listf,
                    "-c", "copy", out], check=True, capture_output=True)
    # cleanup intermediates
    for p in parts:
        os.remove(p)
    os.remove(listf); os.rmdir(TMP)
    dur = subprocess.run([FFMPEG, "-i", out], capture_output=True, text=True).stderr
    import re
    m = re.search(r"Duration: (\S+)", dur)
    print("saved:", out, "| duration", m.group(1) if m else "?")


if __name__ == "__main__":
    main()
