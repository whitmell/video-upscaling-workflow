import os
import shutil
import sys

def move(input_dir, processed_dir, output_dir):
    
    os.makedirs(processed_dir, exist_ok=True)

    # List all images in output dir
    images = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])

    # Move images that are processed
    for image in images:
        source_path = os.path.join(input_dir, image)
        dest_path = os.path.join(processed_dir, image)

        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            print(f"Moved: {image} → {processed_dir}")

