from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import os

app = Flask(__name__, template_folder='templates')

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate one directory up to the 'src' directory
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the 'src' directory to the Python path
sys.path.append(src_dir)

from commands import main as main_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_command', methods=['POST'])
def run_command():
    data = request.get_json()
    command = data['command']
    args = data['args']

    try:
        # Construct the command to run
        output = main_function(command, *args)

        # Return the output
        return jsonify({'output': output, 'error': ""})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)