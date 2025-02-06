import subprocess
import os

def get_audio_offset(video_path):
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=start_time",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    video_start = float(result.stdout.strip())

    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=start_time",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    audio_start = float(result.stdout.strip())

    offset = str(int(1000 * (audio_start - video_start)))
    print(offset)
    return offset

def encode_archive_quality(frame_images_path, original_video_path, output_path, offset):
    command = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-framerate', '59.94',
        '-i', f'{frame_images_path}/frame_%06d.png',
        '-itsoffset', offset,
        '-i', original_video_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'hevc_nvenc',
        '-preset', 'slow',
        '-surfaces', '32',
        '-extra_hw_frames', '32',
        '-rc', 'vbr_hq',
        '-b:v', '50M',
        '-maxrate', '100M',
        '-bufsize', '200M',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        '-rc-lookahead', '32',
        '-vsync', '0',
        '-fps_mode', 'passthrough',
        '-map_metadata', '-1',
        '-map_chapters', '-1',
        '-y', output_path
    ]
    subprocess.run(command, check=True)

def encode_youtube_quality(frame_images_path, original_video_path, output_path, offset):
    command = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-framerate', '59.94',
        '-i', f'{frame_images_path}/frame_%06d.png',
        '-itsoffset', offset,
        '-i', original_video_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'h264_nvenc',
        '-preset', 'p5',
        '-rc', 'vbr',
        '-cq', '22',
        '-b:v', '20M',
        '-maxrate', '40M',
        '-bufsize', '80M',
        '-multipass', 'fullres',
        '-b_ref_mode', 'middle',
        '-rc-lookahead', '32',
        '-spatial_aq', '1',
        '-temporal_aq', '1',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '320k',
        '-fps_mode', 'passthrough',
        '-map_metadata', '-1',
        '-map_chapters', '-1',
        '-y', output_path
    ]
    subprocess.run(command, check=True)

def encode(frame_images_path, original_video_path, output_path, archive_flag=False, youtube_flag=False):
    vidname_without_extension = os.path.splitext(os.path.basename(original_video_path))[0]
    output_archive = os.path.join(output_path, f'{vidname_without_extension}_archive.mkv')
    output_youtube = os.path.join(output_path, f'{vidname_without_extension}_youtube.mp4')
    offset = get_audio_offset(original_video_path)

    if archive_flag:
        encode_archive_quality(frame_images_path, original_video_path, output_archive, offset)
    if youtube_flag:
        encode_youtube_quality(frame_images_path, original_video_path, output_youtube, offset)