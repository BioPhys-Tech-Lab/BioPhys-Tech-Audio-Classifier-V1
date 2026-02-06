
import torch
import os
import argparse
from src.model import BioAcousticModel
from src.preprocessor import process_audio_phd_level

def predict(audio_path):
    print(f"--- Probing Model with {audio_path} ---")
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Load Model
    num_classes = 50
    model = BioAcousticModel(num_classes=num_classes).to(device)
    
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint) # Assuming checkpoint is just state_dict or has 'model_state_dict'
        print("Model loaded successfully.")
    else:
        print("Warning: Checkpoint not found. Initializing with Random Weights.")
    
    model.eval()

    # 3. Preprocess Audio
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} not found.")
        return

    try:
        # returns (128, Time) numpy array
        spectrogram, sr = process_audio_phd_level(audio_path)
        print(f"Spectrogram generated: {spectrogram.shape}")
        
        # Prepare for model: [Batch, 1, Freq, Time]
        # Spectrogram is [Freq, Time] -> need [1, 1, Freq, Time]
        input_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)
        print(f"Input Tensor Shape: {input_tensor.shape}")

        # 4. Inference
        with torch.no_grad():
            logits, attention_weights = model(input_tensor)
            
        print(f"Logits Shape: {logits.shape}")
        print(f"Attention Weights Shape: {attention_weights.shape}")
        
        # 5. Top Predictions
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        print("\nTop 5 Predictions (Class Indices):")
        for i in range(5):
            print(f"Rank {i+1}: Class {top5_catid[0][i].item()} | Confidence: {top5_prob[0][i].item():.4f}")

        print("\n--- Inference Complete ---")

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Default test file from dataset
    DEFAULT_AUDIO = 'dataset/audio/1-100032-A-0.wav'
    
    parser = argparse.ArgumentParser(description='Bioacoustics Probe')
    parser.add_argument('--audio', type=str, default=DEFAULT_AUDIO, help='Path to audio file')
    
    args = parser.parse_args()
    predict(args.audio)
