import os
import glob
import sys
from archive.move_processed import move
from chapters.extract_chapters import extract_chapters
from frame_extraction.extract_frames import extract_frames
from upscaler import upscale_spandrel, FrameDataset
from encoding.encoder import encode

models = {
    "RealESRGAN_x4plus.pth": "D:\\Video\\Models\\upscale\\RealESRGAN_x4plus.pth",
    "BSRGAN.pth": "D:\\Video\\Models\\upscale\\BSRGAN.pth"
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
        if len(sys.argv) < 3:
            print("Usage: python main.py frames <video_file>")
            sys.exit(1)
        input_file = sys.argv[2]
        extract_frames(input_file)
        print("Done extracting chapter videos!")
    elif command == "encode":
        if len(sys.argv) < 6:
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
        if len(sys.argv) < 6:
            print("Usage: python main.py archive <input_dir> <processed_dir> <output_dir>")
            sys.exit(1)
        input_dir = sys.argv[2]
        processed_dir = sys.argv[3]
        output_dir = sys.argv[4]
        move(input_dir, processed_dir, output_dir)
        print("Done encoding video!")
    else:
        print(f"Unknown command '{command}'")

if __name__ == "__main__":
    main()
