# AviSynth+
AviSynth+ is a tool for video post-production. Its sole purpose in this project is to perform the deinterlacing step as the frames are extracted.

## Installing/configuring

I followed this video for installing AviSynth+, it's about as comprehensive as it gets:
https://www.youtube.com/watch?v=C4PyyQoz6eo

I did not use VirtualDub2, however, and while he doesn't use ffmpeg in this video he does have an older one marked "OUTDATED" in the title that does. The important thing is to make sure the setup of AviSynth+ is done properly and the plugins are in place; as long as ffmpeg is compiled with --enable-avisynth it will take care of the rest.  See the README in the resources/ffmpeg folder for guidance on getting an ffmpeg version with AviSynth enabled.

## The .avs File

AVS files contain commands interpreted by AviSynth+ and can be used as the input video file for ffmpeg commands. The frame_extraction step in this python application will generate the .avs file needed for deinterlacing. Here is an explanation of the commands in that file:

```
SetFilterMTMode("QTGMC", 2)             <-- possibly unnecessary but doesn't hurt
FFmpegSource2("VIDEO_PATH", atrack=1)   <-- loads the video file
ConvertToYV12()                         <-- converts the video to the needed color space
AssumeTFF()                             <-- TFF vs BFF is discussed in the video
#AssumeBFF()                            <--   see below for guidance on testing
QTGMC(preset="Slower", EdiThreads=8)    <-- The actual deinterlacing step
BilinearResize(960,540)                 <-- Make sure the ratio of these numbers match the 
                                        <--   target aspect ratio (4:3, 16:9, etc)
Prefetch(24)                            <-- Prefetches frames for better performance - 
                                        <--   should be at least half of CPU cores
```

## BFF vs TFF

Bottom Field First vs Top Field First.  This will be dependent on the way the source video was interlaced. I've found the easiest way to test is to create an AVS file like this:

```
# test.avs
FFmpegSource2("<video path>", atrack=1) 
AssumeBFF() 
```
Then play it with ffplay (packaged with ffmpeg):

```
ffplay -i test.avs
```

If playback is jittery, try AssumeTFF() instead and see if that smooths out playback.  So far it's been 50/50 between the videos I've upscaled.
