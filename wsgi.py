import json
import subprocess

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/mandelbrot", methods=["POST"])
def mandelbrot():
    data = json.loads(request.data)
    output = subprocess.run(
        [
            "./build/core",
            f"--x_min={data['x_min']}",
            f"--x_max={data['x_max']}",
            f"--y_min={data['y_min']}",
            f"--y_max={data['y_max']}",
        ],
        capture_output=True,
    )
    return {"data": output.stdout.decode("utf-8")}
