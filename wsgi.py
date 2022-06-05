import subprocess

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/mandelbrot")
def mandelbrot():
    output = subprocess.run(["./build/core"], capture_output=True)
    return {"data": output.stdout.decode("utf-8")}
