## Setup

 It is recommended to use a python virtual environment.  You may need to run 

```
pip install virtualenv
```

Then set up the virtual environment and install dependencies.

Windows:
```
cd src
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Linux:
```
cd src
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the application with:
```
python main.py <command> [<args>...]
```

## Command Reference

### Chapters
Splits a video file into individual videos using embedded chapter data. Extracted videos are named using the title in the chapter data and stored in the same directory under the chapters/ subdirectory.

Arguments:
- video_file: path to video file

Example:
```
python main.py chapters "title_00.mkv"
```

### Frames
Extracts PNG frames from a video file and stores them in the same directory under the frames/ subdirectory.

Arguments:
- video_file: path to video file

Example:
```
python main.py frames "C:\Videos\HarryHood.mkv"
```

### Upscale
Applies upscaling model to each frame image in the provided directory and stores the upscaled image with the same name in the provided output directory. 

***Important note:*** This function attempts to handle batches more efficiently by allowing the CPU to batch images to send to the GPU, then allowing the CPU to save the processed images while the GPU works on the next batch.  I've had mixed results comparing the speed of this approach against the ChaiNNer pipeline. Either can be used, but I'd recommend using ChaiNNer for the upscale step as it's an established application that provides better visual feedback as the workflow runs.

Arguments:
- input_dir: directory containing original frames
- output_dir: directory to save upscaled frames

Example:
```
python main.py upscale "C:\Videos\frames" "C:\Videos\upscaled-frames"
```

### Archive
Matching by name, identifies all images in the input direcotry that also exist in output directory and moves them to processing directory. 
Useful when upscaling pipeline needs to be stopped and restarted to prevent re-running the upscale model on images that have already been processed.

Arguments:
- input_dir: directory containing original frames
- processed_dir: directory to move processed frames to
- output_dir: directory containing upscaled frames

Example:
```
python main.py archive "C:\Videos\frames" "C:\Videos\processing" "C:\Videos\upscaled-frames"
```

### Encode
Encodes video from frames. By default, encodes a copy optimized for Youtube and a higher quaility copy for archiving.

Arguments:
- input_frame_dir: directory containing upscaled frames
- input_video_dir: original video (for copying audio stream)
- output_dir: directory for saving encoded videos
- archive_flag: True/False to optionally encode a high quality archive copy
- youtube_flag: True/False to optionally encode a copy optimized for youtube upload

Example:
```
python main.py encode "C:\Videos\upscaled-frames" "C:\Videos\HarryHood.mkv" "C:\Videos\Upscaled" True True
```