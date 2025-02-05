import subprocess

def encode_archive_quality(frame_images_path, original_video_path, output_path):
    command = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-framerate', '59.94',
        '-i', f'{frame_images_path}/frame_%06d.png',
        '-itsoffset', '-0.434',
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

def encode_youtube_quality(frame_images_path, original_video_path, output_path):
    command = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-framerate', '59.94',
        '-i', f'{frame_images_path}/frame_%06d.png',
        '-itsoffset', '-0.434',
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
    if archive_flag:
        encode_archive_quality(frame_images_path, original_video_path, f'{output_path}.mkv')
    if youtube_flag:
        encode_youtube_quality(frame_images_path, original_video_path, f'{output_path}.mp4')