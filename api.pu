import base64, io
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

class DigitRecognizerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = DigitRecognizerNet()
model.load_state_dict(torch.load("digit_recognizer_model.pth", map_location="cpu"))
model.eval()

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Digit API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        b64 = data["image"].split(",")[-1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

        # transparent background ko black bana do (white digit on black)
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        bg.alpha_composite(img)
        img = bg.convert("L").resize((28, 28), Image.BOX)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.1307) / 0.3081
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1).numpy()[0]

        return jsonify({
            "prediction": int(np.argmax(probs)),
            "probabilities": [float(p) for p in probs],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
