import os
import subprocess
import json
import sys
import re

def extract_chapters(video_path, output_path=""):

    if output_path == "":
        output_path = os.path.dirname(video_path) + "/chapters"
    os.makedirs(output_path, exist_ok=True)

    # Get chapter data
    probe_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_chapters",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    print(result.stdout)
    data = json.loads(result.stdout)
    chapters = data.get("chapters", [])

    for c in chapters:
        start_time = float(c.get("start_time", 0))
        end_time = float(c.get("end_time", 0))
        title = c.get("tags", {}).get("title", f"chapter_{start_time}")
        # Clean title for filename
        safe_title = re.sub(r'[\\/*?:"<>|]', '_', title)
        
        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-v", "quiet",
            "-stats",
            "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c", "copy",
            f"{output_path}/{safe_title}.mkv"
        ]
        subprocess.run(ffmpeg_cmd)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_chapters.py <video_file>")
        sys.exit(1)
    extract_chapters(sys.argv[1])