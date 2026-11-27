"""Render the Day 3 video: every scene at 1080p30, then join them into Day3_full.mp4.

    python render_all.py [media_dir]

Needs: manim (pip install manim), ffmpeg, and the Pango/Cairo system libraries.
Narration is recorded separately; the scene lengths follow the script timestamps (2:57 total).
"""
import os
import subprocess
import sys

SCENES = ["S1_Hook", "S2_Build", "S3_Data", "S4_Model", "S5_Loss", "S6_Wrap"]
here = os.path.dirname(os.path.abspath(__file__))
media = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(here, "media")

for s in SCENES:
    subprocess.run(["manim", "-r", "1920,1080", "--fps", "30", "--media_dir", media, "day3_video.py", s],
                   cwd=here, check=True)

clips = [os.path.join(media, "videos", "day3_video", "1080p30", f"{s}.mp4") for s in SCENES]
lst = os.path.join(media, "clips.txt")
with open(lst, "w") as f:
    f.writelines(f"file '{c}'\n" for c in clips)
out = os.path.join(media, "Day3_full.mp4")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
print("DONE", out)
