import asyncio
import os
import glob
import sys
from archive.move_processed import move
from chapters.extract_chapters import extract_chapters
from frame_extraction.extract_frames import extract_frames
from upscaler import upscale_spandrel, FrameDataset
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args...]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "upscale":
        if len(sys.argv) < 3:
            print("Usage: python main.py upscale <input_dir> <output_dir>")
            sys.exit(1)

        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        os.makedirs(output_dir, exist_ok=True)

        # upscale_spandrel.process_dir(input_dir, output_dir)
        dataset = FrameDataset(input_dir)
        asyncio.run(upscale_spandrel.process_dataset_async(dataset, output_dir, models["RealESRGAN_x4plus"]))
        upscale_spandrel.process_dataset(dataset, output_dir)
        print("Done upscaling images!")
    elif command == "chapters":
        if len(sys.argv) < 3:
            print("Usage: python main.py chapters <video_file>")
            sys.exit(1)
        input_file = sys.argv[2]
        extract_chapters(input_file)
        print("Done extracting chapter videos!")
    elif command == "frames":
        if len(sys.argv) < 4:
            print("Usage: python main.py frames <video_file> <aspect_ratio>")
            sys.exit(1)
        input_file = sys.argv[2]
        aspect_ratio = sys.argv[3]
        extract_frames(input_file, aspect_ratio)
        print("Done extracting frames!")
    elif command == "encode":
        if len(sys.argv) < 7:
            print("Usage: python main.py encode <input_frame_dir> <input_video_dir> <output_dir> <archive_flag> <youtube_flag>")
            sys.exit(1)
        input_frames = sys.argv[2]
        input_vid = sys.argv[3]
        output_dir = sys.argv[4]
        archive_flag = sys.argv[5]
        youtube_flag = sys.argv[6]
        encode(input_frames, input_vid, output_dir, archive_flag, youtube_flag)
        print("Done encoding video!")
    elif command == "archive":
        if len(sys.argv) < 5:
            print("Usage: python main.py archive <input_dir> <processed_dir> <output_dir>")
            sys.exit(1)
        input_dir = sys.argv[2]
        processed_dir = sys.argv[3]
        output_dir = sys.argv[4]
        move(input_dir, processed_dir, output_dir)
        print("Done archiving processed frames!")
    else:
        print(f"Unknown command '{command}'")

if __name__ == "__main__":
    main()
