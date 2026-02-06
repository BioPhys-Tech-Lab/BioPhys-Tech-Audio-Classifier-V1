import librosa
import librosa.display
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt

def process_audio_phd_level(file_path):
    """
    Process audio with PhD-level purification:
    1. Load at 32kHz (Bioacoustics standard)
    2. Active Noise Reduction (Spectral Gating)
    3. PCEN (Per-Channel Energy Normalization)
    """
    # 1. Cargar el audio (Sampling rate de 32kHz es estándar en bioacústica)
    y, sr = librosa.load(file_path, sr=32000)
    
    # 2. Separación de Fuentes (Source Separation)
    # Usamos HPSS (Harmonic-Percussive Source Separation) como técnica base
    # Harmonic = Biophony (cantos tonales)
    # Percussive = Geophony/Anthropophony a menudo (lluvia, pasos, clicks)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # 3. Spectral Subtraction (Reducción de Ruido)
    # Aplicamos spectral gating sobre la componente armónica para limpiar más
    y_clean = nr.reduce_noise(y=y_harmonic, sr=sr, prop_decrease=0.8, n_std_thresh_stationary=1.5)
    
    # 4. Aplicar PCEN (Per-Channel Energy Normalization)
    S = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_mels=128, fmax=sr/2)
    
    # PCEN normalización dinámica
    pcen_S = librosa.pcen(S * (2**31)) 
    
    return pcen_S, sr

# Ejemplo de visualización para tu demo en Geología
def save_pro_spectrogram(pcen_S, output_path):
    plt.figure(figsize=(10, 4))
    # Note: pcen_S is already kind of log-like, but for visualization standard display usually expects dB or similar range.
    # PCEN output can be visualized directly, but let's check standard librosa visualization.
    librosa.display.specshow(pcen_S, sr=32000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('BioPhys-Tech Lab: High-Res PCEN Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
