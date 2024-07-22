import sys
from pathlib import Path

sys.path.append(Path("..").resolve().as_posix())

import torch

from flask import Flask, request, jsonify

app = Flask(__name__)


CHECKPOINT_PATH = "path/to/checkpoint"


@app.route("/predict_image", methods=["POST"])
def predict_image():
    json_input = request.get_json(force=True)
    output = None
    return jsonify({"prediction": output})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Listen on all interfaces
