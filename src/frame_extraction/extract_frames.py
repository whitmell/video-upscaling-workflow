import os
import subprocess
import sys

template_path="frame_extraction\\templates\\"

def get_video_width(video_path):
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    width = int(result.stdout.strip())

    print(f"Video width: {width}")
    return width

def create_avs(video_path, aspect_ratio):
    vid_dir = os.path.dirname(video_path)
    avs_path = f"{vid_dir}\\deinterlace.avs"
    with open(f"{template_path}default.avs", "r") as f:
        avs_content = f.read()

    avs_content = avs_content.replace("VIDEO_PATH", video_path)

    width = get_video_width(video_path)
    aspect_width, aspect_height = map(int, aspect_ratio.split(':'))
    height = int(width * aspect_height / aspect_width)
    print(f"Setting height to {height}")

    avs_content = avs_content.replace("WIDTH", str(width))
    avs_content = avs_content.replace("HEIGHT", str(height))

    with open(avs_path, "w") as f:
        f.write(avs_content)

    return avs_path

def extract_frames(video_path, aspect_ratio):
    vid_dir = os.path.dirname(video_path)
    frame_dir = f"{vid_dir}\\frames"
    os.makedirs(frame_dir, exist_ok=True)
    # Get chapter data
    probe_cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-i", f"{create_avs(video_path, aspect_ratio)}",
        frame_dir + "\\frame_%06d.png"
    ]
    print(probe_cmd)
    result = subprocess.run(probe_cmd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video_file>")
        sys.exit(1)
    extract_frames(sys.argv[1])