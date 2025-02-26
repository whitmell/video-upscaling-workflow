import asyncio
import os
import glob
import sys
from archive.move_processed import move, move_processed
from chapters.extract_chapters import extract_chapters
from frame_extraction.extract_frames import extract_frames
from upscaler import upscale_spandrel, chainner, FrameDataset
from encoding.encoder import encode

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate one directory up
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Construct the path to the desired file
realesrgan_path = os.path.join(parent_dir, "resources", "models", "upscaling", "RealESRGAN_x4plus.pth")
bsrgan_path = os.path.join(parent_dir, "resources", "models", "upscaling", "BSRGAN.pth")

models = {
    "RealESRGAN_x4plus": realesrgan_path,
    "BSRGAN": bsrgan_path
}

def main(command, *args):
    if command == "upscale":
        if len(args) < 2:
            return "Usage: python main.py upscale <input_dir> <output_dir>"

        input_dir = args[0]
        output_dir = args[1]
        os.makedirs(output_dir, exist_ok=True)

        # upscale_spandrel.process_dir(input_dir, output_dir)
        dataset = FrameDataset(input_dir)
        asyncio.run(upscale_spandrel.process_dataset_async(dataset, output_dir, models["RealESRGAN_x4plus"], move=True))
        upscale_spandrel.process_dataset(dataset, output_dir)
        return "Done upscaling images!"
    elif command == "chainner":
        if len(args) < 2:
            return "Usage: python main.py chainner <input_dir> <output_dir>"
        print("Upscaling with ChaiNNer")
        input_dir = args[0]
        output_dir = args[1]
        os.makedirs(output_dir, exist_ok=True)
        chainner.upscale(dataset, output_dir)
        return "Done upscaling images!"
    elif command == "chapters":
        if len(args) < 1:
            return "Usage: python main.py chapters <video_file>"
        input_file = args[0]
        extract_chapters(input_file)
        return "Done extracting chapter videos!"
    elif command == "frames":
        if len(args) < 2:
            return "Usage: python main.py frames <video_file> <aspect_ratio>"
        input_file = args[0]
        aspect_ratio = args[1]
        extract_frames(input_file, aspect_ratio)
        return "Done extracting frames!"
    elif command == "encode":
        print(args)
        if len(args) < 5:
            return "Usage: python main.py encode <input_frame_dir> <input_video_dir> <output_dir> <archive_flag> <youtube_flag>"
        input_frames = args[0]
        input_vid = args[1]
        output_dir = args[2]
        archive_flag = args[3]
        youtube_flag = args[4]
        encode(input_frames, input_vid, output_dir, archive_flag, youtube_flag)
        return "Done encoding video!"
    elif command == "archive":
        if len(args) < 3:
            return "Usage: python main.py archive <input_dir> <processed_dir> <output_dir>"
        input_dir = args[0]
        processed_dir = args[1]
        output_dir = args[2]
        move_processed(input_dir, processed_dir, output_dir)
        return "Done archiving processed frames!"
    else:
        return f"Unknown command '{command}'"

if __name__ == "__main__":
    command = sys.argv[1]
    result = main(command, *sys.argv[2:])
    print(result)
