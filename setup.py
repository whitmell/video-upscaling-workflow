import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Construct the path to the requirements.txt file located in src/
# requirements_path = os.path.join(here, 'src', 'requirements.txt')

# # Read the requirements.txt file and parse each line
# with open(requirements_path, 'r') as f:
#     requirements = [
#         line.strip() for line in f
#         if line.strip() and not line.startswith('#')
#     ]

setup(
    name='video-upscaling-workflow',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where=f"{os.path.join(here, 'src')}"),
    install_requires=[
        'asyncio',
        'numpy',
        'pillow==11.0.0',
        'spandrel',
        'fastapi',
        'pydantic',
        'redis',
        'rq',
        'tqdm==4.66.5',
        'Flask[async]',
        'gunicorn',
        'uvicorn[standard]',
    ],
    entry_points={
        'console_scripts': [
            'runvid = app.routes:run_app',
        ],
    },
)