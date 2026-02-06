import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import requests
import json
from pathlib import Path
import numpy as np
import random

# Import our PhD-level preprocessor
# Assuming this script is run from project root, or src is in path
try:
    from src.preprocessor import process_audio_phd_level
except ImportError:
    from preprocessor import process_audio_phd_level

class BioDataset(Dataset):
    """
    Bioacoustic Dataset Loader.
    Capable of handling metadata CSVs (like iNaturalist or ESC-50).
    """
    def __init__(self, metadata_file, audio_root, transform=None, augment=False, cache_dir=None):
        """
        Args:
            metadata_file (str): Path to CSV/XLSX with filenames and labels.
            audio_root (str): Directory containing audio files.
            transform (callable, optional): Optional transform to be applied.
        """
        self.audio_root = audio_root
        self.audio_root = audio_root
        self.transform = transform
        self.augment = augment
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load metadata
        if metadata_file.endswith('.csv'):
            self.df = pd.read_csv(metadata_file)
        elif metadata_file.endswith('.xlsx'):
            self.df = pd.read_excel(metadata_file)
        else:
            raise ValueError("Unsupported metadata format")
            
        # Basic standardization of column names if needed
        # We assume 'filename' and 'target' or 'category' exist
        self.filename_col = 'filename' if 'filename' in self.df.columns else self.df.columns[0]
        self.label_col = 'category' if 'category' in self.df.columns else 'target'
        
        # Create label mapping if targets are strings
        if self.df[self.label_col].dtype == object:
            self.labels = self.df[self.label_col].unique()
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        else:
            self.label_to_idx = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_name = row[self.filename_col]
        audio_path = os.path.join(self.audio_root, audio_name)
        
        # --- PhD Level Preprocessing ---
        # spectogram shape: [Mels, Time]
        spectrogram = None
        sr = 32000 # Default
        
        # --- Caching Logic ---
        if self.cache_dir:
             spectrogram = self._load_cache(audio_name)
             
        if spectrogram is None:
            # --- PhD Level Preprocessing ---
            # Returns (Spectrogram, SampleRate)
            spectrogram, sr = process_audio_phd_level(audio_path)
            
            if self.cache_dir:
                self._save_cache(audio_name, spectrogram)
        
        # --- Data Augmentation (PhD Level) ---
        if self.augment:
             spectrogram = self._augment_spectrogram(spectrogram)
        
        # Convert to Tensor [Channels, Freq, Time]
        # We add a channel dimension for the CNN (EfficientNet expects input channels)
        spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()
        
        # Get Label
        label = row[self.label_col]
        if self.label_to_idx:
            label = self.label_to_idx[label]
        
        label_tensor = torch.tensor(label).long()
        
        return spectrogram_tensor, label_tensor

    def _augment_spectrogram(self, spectrogram):
        """
        Apply SpecAugment, Time Shift, and Noise Injection
        spectrogram: numpy array [Mels, Time]
        """
        # 1. Time Shift (Rolling)
        if random.random() < 0.5:
            shift = random.randint(0, int(spectrogram.shape[1] * 0.2)) # Shift up to 20%
            spectrogram = np.roll(spectrogram, shift, axis=1)

        # 2. Gaussian Noise Injection
        if random.random() < 0.5:
             noise_level = np.random.uniform(0, 0.05)
             noise = np.random.randn(*spectrogram.shape) * noise_level * np.max(spectrogram)
             spectrogram = spectrogram + noise

        # 3. SpecAugment: Frequency Masking
        if random.random() < 0.5:
            f_mask_param = 15 # Mask up to 15 mel bands
            f0 = random.randint(0, max(0, spectrogram.shape[0] - f_mask_param))
            spectrogram[f0:f0+f_mask_param, :] = 0

        # 4. SpecAugment: Time Masking
        if random.random() < 0.5:
            t_mask_param = 30 # Mask up to 30 frames
            t0 = random.randint(0, max(0, spectrogram.shape[1] - t_mask_param))
            spectrogram[:, t0:t0+t_mask_param] = 0
            
        return spectrogram

    def _get_cache_path(self, filename):
        # Create a safe filename for cache (replace / with _)
        safe_name = filename.replace('/', '_').replace('\\', '_') + '.npy'
        return os.path.join(self.cache_dir, safe_name)

    def _load_cache(self, filename):
        path = self._get_cache_path(filename)
        if os.path.exists(path):
            try:
                return np.load(path)
            except:
                return None
        return None

    def _save_cache(self, filename, spectrogram):
        path = self._get_cache_path(filename)
        np.save(path, spectrogram)

class XenoCantoDownloader:
    """
    Tool for fusing datasets with expert data from Xeno-canto.
    """
    BASE_URL = "https://www.xeno-canto.org/api/2/recordings"
    
    def __init__(self, download_path='dataset/xeno_canto'):
        self.download_path = download_path
        os.makedirs(self.download_path, exist_ok=True)
        
    def search_species(self, query):
        """
        Search for species (e.g., 'cnt:Peru gen:Ara')
        """
        params = {'query': query}
        response = requests.get(self.BASE_URL, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data: {response.status_code}")
            return None
            
    def download_recordings(self, query, max_files=10):
        print(f"Searching Xeno-canto for: {query}")
        data = self.search_species(query)
        if not data or 'recordings' not in data:
            print("No recordings found.")
            return
        
        recordings = data['recordings']
        print(f"Found {len(recordings)} recordings. Downloading top {max_files}...")
        
        count = 0
        meta_data = []
        
        for rec in recordings[:max_files]:
            file_url = rec['file']
            file_name = f"{rec['id']}_{rec['gen']}-{rec['sp']}.mp3"
            save_path = os.path.join(self.download_path, file_name)
            
            try:
                # Download audio
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                # Keep metadata
                meta_data.append({
                    'filename': file_name,
                    'category': f"{rec['gen']} {rec['sp']}",
                    'id': rec['id'],
                    'country': rec['cnt']
                })
                count += 1
                print(f"Downloaded: {file_name}")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
                
        # Save Metadata for this batch
        df = pd.DataFrame(meta_data)
        csv_path = os.path.join(self.download_path, 'xeno_metadata.csv')
        # Append if exists
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
            
        print(f"Download complete. {count} files saved to {self.download_path}")

if __name__ == "__main__":
    # Test Block
    # 1. Test Xeno-canto
    downloader = XenoCantoDownloader(download_path='dataset/xeno_test')
    # Example: Search for a common Andean bird if you want, or just generic
    # downloader.download_recordings(query='cnt:Peru pmo:A', max_files=1) 
    
    # 2. Test Dataset Loader (Assuming ESC-50 exists from user context)
    # dataset = BioDataset(metadata_file='dataset/meta/esc50.csv', audio_root='dataset/audio')
    # print(f"Dataset length: {len(dataset)}")
    pass
