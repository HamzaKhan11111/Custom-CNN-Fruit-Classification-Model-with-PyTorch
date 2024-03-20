from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = nn.Sequential(
    nn.Conv2d(3, 6, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 53 * 53, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 131)
)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Define transformations for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming input size of the model is 224x224
    transforms.ToTensor(),
])

# Function to predict the class of an image
def predict_image_class(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Endpoint for predicting image class
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    prediction = predict_image_class(image)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
