
import streamlit as st
import torch
import torch.nn as nn
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

# 1. Model Structure (Must match trained model)
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
        x = self.pool(self.relu(self.conv1(x))) # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x))) # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model Load krna
@st.cache_resource
def load_model():
    model = DigitRecognizerNet()
    model.load_state_dict(torch.load('digit_recognizer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")

st.title("✏️ Handwritten Digit Recognition (Scratch Model)")
st.write("Canvas par koi bhi number (0-9) likhein aur system predict kray ga!")

# Draw krne ke liye Canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Get drawing image
    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    
    # Check if user has drawn anything
    if np.sum(img) > 0:
        # Resize to 28x28 like MNIST dataset
        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Standard Normalization
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = (img_tensor - 0.1307) / 0.3081
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).unsqueeze(0) # Batch size 1, Channel 1
        
        with torch.no_grad():
            outputs = model(img_tensor)
            prediction = torch.argmax(outputs, 1).item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            
        st.write(f"## Predicted Digit: **{prediction}**")
        st.bar_chart(probabilities)
