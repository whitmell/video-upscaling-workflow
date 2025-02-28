# main.py
from contextlib import asynccontextmanager
import os
import asyncio
from tqdm import tqdm
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
from commands import main as main_function
from upscaler.upscale_batch import BatchUpscaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Environment configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WORKER_COUNT = int(os.getenv("WORKER_COUNT", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

# This Pydantic model enforces the JSON structure
class CommandRequest(BaseModel):
    command: str
    args: List[str] = []

# Global upscaler instance to ensure model is loaded only once
upscaler = None

def close_progress_bar(progress_bar: tqdm, last_batch: bool = False):
    if last_batch:
        progress_bar.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the upscaler model
    global upscaler
    print("Initializing upscaler...")
    upscaler = BatchUpscaler()
    print("Upscaler initialized successfully")
    
    yield

    # Clean up resources
    print("Shutting down...")
    if upscaler:
        upscaler.unload_model()


app = FastAPI(lifespan=lifespan)

# Background task storage
background_tasks = {}
task_id_counter = 0

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    # Serve template/index.html
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run_command")
async def run_command(request_data: CommandRequest, background_tasks: BackgroundTasks):
    """Run a command based on the request data and return immediately."""
    global task_id_counter
    command = request_data.command
    args = request_data.args
    
    # Generate a unique task ID
    task_id = task_id_counter
    task_id_counter += 1
    
    response = {
        "status": "started",
        "message": f"Command '{command}' started",
        "task_id": task_id
    }
    
    if command == "upscale" and len(args) >= 2:
        # Configure upscaling parameters
        input_dir = args[0]
        output_dir = args[1]
        move = True
        
        if not upscaler:
            return {"error": "Upscaler not initialized"}
        
        # Start upscaling in the background
        background_tasks.add_task(
            run_upscale_task, 
            upscaler=upscaler, 
            input_dir=input_dir, 
            output_dir=output_dir, 
            move=move, 
            batch_size=BATCH_SIZE,
            task_id=task_id
        )
        
    else:
        # Start other commands in background
        background_tasks.add_task(
            run_command_task,
            command=command,
            args=args,
            task_id=task_id
        )
    
    return response

async def run_upscale_task(upscaler, input_dir, output_dir, move, batch_size, task_id):
    """Run upscale task in the background"""
    try:
        result = await upscaler.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            move=move,
            batch_size=batch_size
        )
        background_tasks[task_id] = result
    except Exception as e:
        background_tasks[task_id] = {"error": str(e)}

def run_command_task(command, args, task_id):
    """Run other commands in the background"""
    try:
        output = main_function(command, *args)
        background_tasks[task_id] = {"output": output, "status": "completed"}
    except Exception as e:
        background_tasks[task_id] = {"error": str(e), "status": "error"}

@app.get("/status")
async def get_status():
    """Get the current processing status of the upscaler"""
    return upscaler.get_status() if upscaler else {"status": "no_upscaler", "message": "Upscaler not initialized"}
