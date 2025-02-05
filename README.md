# Video Upscaling Workflow

This repo contains a python app with a handful of functions to assist in upscaling videos.

## Prerequisites

### CUDA
It is recommended to install CUDA to get the most out of this workflow, as it will enable GPU processing and greatly speed up most of the steps. Some modifications may need to be made to the code to run without cuda (likely just removing the instances of .cuda() found in upscaler and -hwaccel cuda in ffmpeg commands)

You can find instructions for installing CUDA for your environment here: https://docs.nvidia.com/cuda/cuda-quick-start-guide/
For this workflow you just need your updated NVIDIA driver and the CUDA Toolkit.  This application was tested on Windows 11 running CUDA 12.8, with a NVIDIA GTX 4070 Laptop GPU.

### A Note on WSL2
If you plan on running this application in a WSL2 instance, you will need to carefully follow the guide here to set up CUDA: https://docs.nvidia.com/cuda/cuda-quick-start-guide/#wsl

You should **not** install NVIDIA's display driver!  This will conflict with the CUDA drivers and you'll need to uninstall both and start over.
After successfully installing CUDA and verifying that it recognized my GPU with "nvidia-smi", I was able to perform *most* of the steps of this workflow, with the one exception of the deinterlacing step using QTGMC during frame extraction. QTGMC relies on OpenCL, which is not yet supported on WSL2, and even after installing POCL, an alternative that supposedly works on WSL2, I eventually exhausted my efforts at running this step in my WSL2 Ubuntu instance. Hopefully this will improve in the future as 

I did experience **better performance upscaling from WSL2**, but running the upscaler from WSL2 against frame files in the Windows filesystem is not recommended; it **will** be slow. You could extract the frames in Windows and then move the frames folder to WSL2 for upscaling (I'd recommend compressing to 7z first), it's just a bit more of a hassle.

## Resources

Files and scripts which are needed for steps in this workflow can be found in the resources/ folder.  Each subfolder has a README of its own with useful links and tips.

- AviSynth+: tool for video post-production, used in this workflow for deinterlacing.  [Read More](./resources/avisynth/README.md)
- ChaiNNer: node-based image processing GUI, can be used for batch upscaling instead of the "upscale" command in the python app.  [Read More](./resources/chainner/README.md)
- ffmpeg: powerful command-line video processing tool, used in most steps of this workflow. [Read More](./resources/ffmpeg/README.md)
- models: this folder contains the AI models used in the upscaling step. [Read More](./resources/models/README.md)

## Python app

Found in the /src folder

See the app documentation for setup and command reference: [Doc](./src/README.md)