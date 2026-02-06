import torch
from torch.utils.data import DataLoader
from src.model import BioAcousticModel
from src.data_loader import BioDataset
import os

def main():
    print("=== PhD Bioacoustics Pipeline Initialized ===")
    
    # 1. Setup Configuration
    # Adjust these paths to your actual dataset locations
    META_FILE = 'dataset/meta/esc50.csv'
    AUDIO_ROOT = 'dataset/audio'
    
    # Check if dataset exists to run demo
    if not os.path.exists(META_FILE):
        print(f"Warning: Dataset metadata not found at {META_FILE}. Please ensure dataset is correctly placed.")
        print("Skipping real data loading for demo...")
        has_data = False
    else:
        has_data = True

    # 2. Initialize Model (Backbone: EfficientNet + Transformer + Attention)
    print("\n--- Initializing Hybrid Model ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 50 classes for ESC-50, change as needed for iNaturalist/Xeno-canto
    num_classes = 50 
    model = BioAcousticModel(num_classes=num_classes).to(device)
    print("Model loaded successfully.")
    
    # 3. Data Loading & Transform Check
    if has_data:
        print("\n--- Testing Data Pipeline ---")
        try:
            dataset = BioDataset(metadata_file=META_FILE, audio_root=AUDIO_ROOT)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            # Fetch one batch
            spectrograms, labels = next(iter(dataloader))
            spectrograms = spectrograms.to(device)
            
            print(f"Input Batch Shape: {spectrograms.shape}") 
            # Expected: [Batch, 1, Mels, Time] e.g. [2, 1, 128, ~Time]
            
            # 4. Forward Pass
            logits, attention_weights = model(spectrograms)
            
            print(f"Logits Shape: {logits.shape}") # [2, 50]
            print(f"Attention Weights Shape: {attention_weights.shape}") # [2, SeqLen, 1]
            print("Forward pass successful!")
            
        except Exception as e:
            print(f"Error during data pipeline test: {e}")
    else:
        print("\n[!] No dataset found. Running with Synthetic Data to verify Architecture.")
        # Synthetic Test
        dummy_input = torch.randn(2, 1, 128, 500).to(device)
        logits, _ = model(dummy_input)
        print(f"Synthetic Forward Pass Logits: {logits.shape}")

if __name__ == "__main__":
    main()
