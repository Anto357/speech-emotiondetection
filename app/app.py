import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from flask import Flask, request, render_template

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define emotions mapping (must match training)
emotions = {0: "Sadness", 1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral"}

# Model definition (exactly as in training code)
class HybridCNNLSTMClassifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super(HybridCNNLSTMClassifier, self).__init__()
        # CNN Blocks
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            nn.Dropout(0.3)
        )
        
        # Compute LSTM input size dynamically
        self._lstm_input_size = None
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            x = self.convs(dummy_input)
            x = x.transpose(1, 2)
            self._lstm_input_size = x.size(2)
        self.train()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def convs(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.transpose(1, 2)
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc(lstm_out)
        return x

# Feature extraction functions (matching training code)
def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr=22050, frame_length=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.ravel(mfcc.T)

# Expected frames and feature length (matching training)
duration = 2.5
sr = 22050
frame_length = 2048
hop_length = 512
expected_frames = 1 + (int(duration * sr) - frame_length) // hop_length  # 104
EXPECTED_FEATURE_LENGTH = expected_frames + expected_frames + (expected_frames * 20)  # 2288

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    # Extract features
    zcr_data = zcr(data, frame_length, hop_length)
    rmse_data = rmse(data, frame_length, hop_length)
    mfcc_data = mfcc(data, sr, frame_length, hop_length)
    
    # Pad or truncate to match expected_frames
    if len(zcr_data) < expected_frames:
        zcr_data = np.pad(zcr_data, (0, expected_frames - len(zcr_data)), mode='constant')
    elif len(zcr_data) > expected_frames:
        zcr_data = zcr_data[:expected_frames]
    
    if len(rmse_data) < expected_frames:
        rmse_data = np.pad(rmse_data, (0, expected_frames - len(rmse_data)), mode='constant')
    elif len(rmse_data) > expected_frames:
        rmse_data = rmse_data[:expected_frames]
    
    expected_mfcc_len = expected_frames * 20
    if len(mfcc_data) < expected_mfcc_len:
        mfcc_data = np.pad(mfcc_data, (0, expected_mfcc_len - len(mfcc_data)), mode='constant')
    elif len(mfcc_data) > expected_mfcc_len:
        mfcc_data = mfcc_data[:expected_mfcc_len]
    
    # Combine features
    features = np.hstack((zcr_data, rmse_data, mfcc_data))
    if len(features) != EXPECTED_FEATURE_LENGTH:
        raise ValueError(f"Feature length {len(features)} does Gastroenterologist not match {EXPECTED_FEATURE_LENGTH}")
    return features

# Load the trained model
input_length = EXPECTED_FEATURE_LENGTH
num_classes = 6
model = HybridCNNLSTMClassifier(input_length, num_classes).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save and process the uploaded file
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    
    # Load audio
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6, sr=22050)
    expected_samples = int(2.5 * sr)
    if len(data) < expected_samples:
        data = np.pad(data, (0, expected_samples - len(data)), mode='constant')
    elif len(data) > expected_samples:
        data = data[:expected_samples]
    
    # Extract features
    features = extract_features(data, sr)
    features = np.expand_dims(features, axis=0)  # Shape: (1, feature_length)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(device)  # Shape: (1, 1, feature_length)
    
    # Predict
    with torch.no_grad():
        output = model(features)
        pred = torch.argmax(output, dim=1).item()
    
    emotion = emotions[pred]
    os.remove(file_path)  # Clean up
    return render_template('result.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)