import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import os
import time

# Import project modules
from src.model import BioAcousticModel
from src.data_loader import BioDataset

def train_model(
    meta_file='dataset/meta/esc50.csv', 
    audio_root='dataset/audio',
    num_epochs=10, 
    batch_size=16, 
    learning_rate=1e-4,
    save_dir='checkpoints'
):
    print("=== Starting Training Pipeline (PhD Standard) ===")
    
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Prepare Data
    print("Loading Dataset...")
    print("Loading Dataset...")
    # Load base dataset to get metadata
    base_dataset = BioDataset(metadata_file=meta_file, audio_root=audio_root)
    
    # Split Train/Val (80/20 standard)
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    
    # Get indices for split
    train_subset_indices, val_subset_indices = random_split(base_dataset, [train_size, val_size])
    
    # Create distinct datasets: Train (Augmented) vs Val (Clean)
    train_ds_aug = BioDataset(metadata_file=meta_file, audio_root=audio_root, augment=True, cache_dir='dataset/cache')
    val_ds_clean = BioDataset(metadata_file=meta_file, audio_root=audio_root, augment=False, cache_dir='dataset/cache')
    
    # Create final Subsets using the indices
    train_dataset = Subset(train_ds_aug, train_subset_indices.indices)
    val_dataset = Subset(val_ds_clean, val_subset_indices.indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # workers=0 for Win safety
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    # 3. Initialize Model
    # Determine num_classes automatically if possible, else 50 (ESC-50)
    # Determine num_classes from base_dataset
    num_classes = len(base_dataset.labels) if hasattr(base_dataset, 'labels') and base_dataset.labels is not None else 50
    print(f"Initializing Model for {num_classes} classes...")
    
    model = BioAcousticModel(num_classes=num_classes).to(device)
    
    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 5. Training Loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # --- Training Phase ---
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs, _ = model(inputs) # Ignore attention weights for loss
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Completed in {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:  {val_loss:.4f} | Val Acc:  {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Model Saved: {val_acc:.2f}%")
            
    print("Training Complete.")

if __name__ == "__main__":
    # Example usage
    train_model()
