# main.py
from contextlib import asynccontextmanager
import os
from tqdm import tqdm
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
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

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    # Serve template/index.html
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run_command")
async def run_command(request_data: CommandRequest):
    """Run a command based on the request data."""
    command = request_data.command
    args = request_data.args
    
    if command == "upscale" and len(args) >= 2:
        # Direct upscaling without queues or workers
        input_dir = args[0]
        output_dir = args[1]
        move = True
        
        # Process directory directly with the preloaded upscaler
        if upscaler:
            result = await upscaler.process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                move=move,
                batch_size=BATCH_SIZE
            )
            return JSONResponse(content=result)
        else:
            return {"error": "Upscaler not initialized"}
    else:
        # Use the main_function for other commands
        try:
            output = main_function(command, *args)
            return {"output": output}
        except Exception as e:
            return {"error": str(e)}
