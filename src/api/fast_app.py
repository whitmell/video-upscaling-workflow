# main.py
from contextlib import asynccontextmanager
import glob
import os
import subprocess
import multiprocessing
from tqdm import tqdm
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from redis import Redis
from rq import Queue, Worker
from commands import main as main_function
from upscaler.upscale_batch import process_batch_async

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Environment configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WORKER_COUNT = int(os.getenv("WORKER_COUNT", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

# Set up Redis connection
redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT)

# Define two queues: primary and secondary
command_queue = Queue("command", connection=redis_conn)
batch_queue = Queue("batch", connection=redis_conn)


# This Pydantic model enforces the JSON structure
class CommandRequest(BaseModel):
    command: str
    args: List[str] = []


# Background task that may queue additional tasks on the secondary queue
def run_command_in_background(command: str, args: List[str]):
    if command == "upscale":
        run_batch_upscale(command, args)
    else:
        main_function(command, *args)
        
def close_progress_bar(progress_bar: tqdm, last_batch: bool = False):
    if last_batch:
        progress_bar.close()

def run_batch_upscale(command: str, args: List[str]):
    input_path = args[0]
    output_path = args[1]
    
    # Get all files in the input path
    files = glob.glob(os.path.join(input_path, '*'))
    total_files = len(files)
    batches = [files[i:i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    total_batches = len(batches)
    batch_num = 1

    
    print(f"Starting batch processing of {total_files} files...")
    progress_bar = tqdm(total=total_files, desc="Processing images")

    for batch in batches:
        batch_queue.enqueue(process_batch, batch, output_path, batch_num, total_batches, progress_bar, close_progress_bar)
        batch_num += 1

def process_batch(batch: List[str], output_path: str, batch_num: int, total_batches: int, progress_bar: tqdm, hook_function=None):
    print(f"Processing batch: {batch_num} of {total_batches}")
    try:
        for file in batch:
        # Call the main function for each file in the batch
            process_batch_async(batch, output_path, move=True)
            progress_bar.update(1)
            # Call the hook function if provided
            if hook_function:
                hook_function(progress_bar, batch_num == total_batches)
        print(f"Finished batch {batch_num}")
    except Exception as e:
        print(f"Error processing batch {batch_num}: {e}")
        batch_queue.enqueue(process_batch, batch, output_path, batch_num, total_batches, progress_bar, close_progress_bar)

# A simple secondary task (could do anything you like here)
def run_secondary_task(command: str, args: List[str]):
    subprocess.run([command, *args])

def start_worker(queue_name):
    worker = Worker([queue_name], connection=redis_conn)
    worker.work()

@asynccontextmanager
async def lifespan(app: FastAPI):
    for queue_name in ["command", "batch"]:
        for _ in range(WORKER_COUNT):
            # Use ProcessPoolExecutor to run workers in separate processes
            process = multiprocessing.Process(
                target=start_worker,
                args=(queue_name,),
                daemon=True  # Make workers daemon processes so they stop when the main process stops
            )
            process.start()
    yield
    print("Shutting down workers...")
    for queue_name in ["command", "batch"]:
        redis_conn.delete(queue_name)


app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    # Serve template/index.html
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run_command")
async def run_command(request_data: CommandRequest):
    # Enqueue the primary command
    command_queue.enqueue(run_command_in_background, request_data.command, request_data.args)
    return {"status": "command queued"}    
