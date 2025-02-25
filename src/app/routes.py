from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import os
import asyncio
from commands import main as main_function

app = Flask(__name__, template_folder='templates')

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate one directory up to the 'src' directory
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the 'src' directory to the Python path
sys.path.append(src_dir)

async def execute_command(command, *args):
    """
    Asynchronously executes the command.
    """
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, main_function, command, *args)
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pause', methods=['GET'])
async def pause():
    # Implement pause logic here
    return jsonify({'status': 'pausing'})

@app.route('/stop', methods=['GET'])
async def stop():
    # Implement stop logic here
    return jsonify({'status': 'stopping'})

@app.route('/run_command', methods=['POST'])
async def run_command():
    data = request.get_json()
    command = data['command']
    args = data['args']

    # Execute the command asynchronously
    asyncio.create_task(execute_command(command, *args))

    # Return immediately with a success message
    return jsonify({'status': 'Command started successfully', 'command': command, 'args': args})

def run_app():
    app.run(debug=True)

if __name__ == '__main__':
    run_app()