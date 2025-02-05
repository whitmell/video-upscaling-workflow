# FFMPEG
FFMPEG is a command-line tool for decoding & encoding video, as well as a wide variety of other functions.  For the purposes of this project a build with --avisynth and --cuda is necessary. It can be built from source or you can download a static build (recommended).

## Static build downloads

For Windows:
https://www.gyan.dev/ffmpeg/builds/ - The Essential build is sufficient for this project

For Linux (and Windows):
https://github.com/BtbN/FFmpeg-Builds - The "gpl" variant is sufficient for this project

## Source code

I *strongly* recommend using a static build rather than building from source, having gone through that process, which required cloning many dependent source repositories and building the libraries individually. However if you're feeling brave:

```
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
```

## FFMPEG Commands used in this project

### Frame extraction

The command below will extract frames from the input_video and place them in the provided frame directory with the name:

frame_xxxxxx.png

The syntax "frame_%06d.png" pads the frame numbers to six digits (i.e. frame_000001.png, frame_000002.png).  Adjust the number as needed if the frame count will exceed 999,999.
```
ffmpeg -hwaccel cuda -i <input_video> <frame_directory>\frame_%06d.png
```

### Encoding Archive Quality Video

```
ffmpeg -hwaccel cuda -framerate 59.94 -i "upscaled/frame_%06d.png" -i "OriginalVideo.mkv" -map 0:v -map 1:a -c:v hevc_nvenc -preset slow -surfaces 32 -extra_hw_frames 32 -rc vbr_hq -b:v 50M -maxrate 100M -bufsize 200M -pix_fmt yuv420p -c:a copy -rc-lookahead 32 -vsync 0 -fps_mode passthrough -map_metadata -1 -map_chapters -1 -y "C:\dev\UpscaledVideo-Archive.mkv"
```

### Encoding Youtube-Optimized Video

```
ffmpeg -hwaccel cuda -framerate 59.94 -i "upscaled/frame_%06d.png" -i "OriginalVideo.mkv" -map 0:v -map 1:a -c:v h264_nvenc -preset p5 -rc vbr -cq 22 -b:v 20M -maxrate 40M -bufsize 80M -multipass fullres -b_ref_mode middle -rc-lookahead 32 -spatial_aq 1 -temporal_aq 1 -pix_fmt yuv420p -c:a aac -b:a 320k -fps_mode passthrough -map_metadata -1 -map_chapters -1 -y "C:\dev\UpscaledVideo-YouTube.mp4"
```

### Fixing audio sync

If the audio in your encoded video is out of sync, there may be an audio offset in your original video. You can use [Mediainfo](https://www.majorgeeks.com/files/details/mediainfo_lite.html) to examine the source audio track. Look for "Delay relative to video". Example:

> Delay relative to video                  : -431 ms

To fix this in encoding, add the following flag in your ffmpeg encode commands, after the input video path arguments:
```
-itsoffset -0.431 
```

The encode command in the python app calculates this value and includes it when it runs the ffmpeg command.