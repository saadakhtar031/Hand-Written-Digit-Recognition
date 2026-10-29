from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Lovable frontend se requests allow karne ke liye

# Model Structure (Must match trained model)
class DigitRecognizerNet(nn.Module):
    def __init__(self):
        super(DigitRecognizerNet, self).__init__()
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
        x = self.fc2(x)
        return x

# Model Load krna
model = DigitRecognizerNet()
try:
    model.load_state_dict(torch.load('digit_recognizer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        
        # Image decode karein
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Resize to 28x28 like MNIST dataset
        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Standard Normalization
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = (img_tensor - 0.1307) / 0.3081
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            prediction = torch.argmax(outputs, 1).item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0].tolist()
            
        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
