import os
import subprocess
import sys

template_path="frame_extraction\\templates\\"

def create_avs(video_path):
    vid_dir = os.path.dirname(video_path)
    avs_path = f"{vid_dir}\\deinterlace.avs"
    with open(f"{template_path}default.avs", "r") as f:
        avs_content = f.read()

    avs_content = avs_content.replace("VIDEO_PATH", video_path)

    with open(avs_path, "w") as f:
        f.write(avs_content)

    return avs_path

def extract_frames(video_path):
    vid_dir = os.path.dirname(video_path)
    frame_dir = f"{vid_dir}\\frames"
    os.makedirs(frame_dir, exist_ok=True)
    # Get chapter data
    probe_cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-i", f"{create_avs(video_path)}",
        frame_dir + "\\frame_%06d.png"
    ]
    print(probe_cmd)
    result = subprocess.run(probe_cmd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video_file>")
        sys.exit(1)
    extract_frames(sys.argv[1])