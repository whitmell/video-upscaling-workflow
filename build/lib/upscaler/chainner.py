import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

CHAINNER_PATH = r"C:\\Users\\whitm\\AppData\\Local\\chaiNNer\\chaiNNer.exe"
CHN_PATH = r"D:\\Video\\video-upscaling-workflow\\src\\upscaler\\chainner.py"

async def upscale(input_dir: str, output_dir: str) -> subprocess.Popen:
    """
    Upscale images using chaiNNer.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory where upscaled images will be saved
        config: Configuration object
        
    Returns:
        subprocess.Popen: The process object for the running upscaler
    """
    if not os.path.exists(CHAINNER_PATH):
        raise FileNotFoundError(f"chaiNNer executable not found at {CHAINNER_PATH}")

    input_overrides = f'{{ "inputs": {{ '
    input_overrides += f'"#0bbed102-e511-4b24-8641-2489ce882a9d:0": "{input_dir}", '
    input_overrides += f'"#4f16fde0-b905-4f63-b74b-e49b72fc192d:1": "{output_dir}" '
    input_overrides += '}}'

    os.makedirs(output_dir, exist_ok=True)
   
    # Save input_overrides to a file
    override_path = os.path.join(output_dir, "input_overrides.json")
    with open(override_path, 'w') as f:
        f.write(input_overrides)
    
    # run "path/to/your-chain.chn" --override "path/to/your-input-overrides.json"
    # Build command
    cmd = [
        CHAINNER_PATH,
        "run", CHAINNER_PATH,
        "--override", input_overrides
    ]

    logger.info(f"Running chaiNNer with command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        return process
        
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to start chaiNNer process: {e}")
        raise

def check_output(process: subprocess.Popen) -> Optional[str]:
    """
    Check the output of the running chaiNNer process.
    
    Args:
        process: The running process to check
        
    Returns:
        Optional[str]: Any error message, or None if process is still running
    """
    if process.poll() is not None:
        _, stderr = process.communicate()
        if process.returncode != 0:
            return f"chaiNNer failed with error: {stderr}"
    return None
