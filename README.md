# CUDA Mandelbrot Julia

A visualizer for the Mandelbrot set and Julia sets using the GPU with CUDA®

## Dependencies

- NVIDIA® CUDA® Toolkit

## Usage

- Compile the core CUDA® code: `make`
- Create a Python virtual environmnent: `python3.10 -m venv .venv --prompt server`
- Activate the virtual environment: `source .venv/bin/activate`
- Install the dependencies: `pip install -r requirements.txt`
- Run the Flask server: `flask run`

## Roadmap

- [x] Change CUDA ouput to Base64 of 8bit uint
- [ ] Display Mandelbrot set
- [ ] Add possiblity of zoom
- [ ] Add julia sets
