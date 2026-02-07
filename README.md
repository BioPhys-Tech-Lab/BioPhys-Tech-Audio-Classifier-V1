# 🧬 BioPhys-Tech Lab: Audio Classifier V1.0

This repository contains the **Minimum Viable Product (MVP)** for automated bioacoustic classification. The system utilizes a cutting-edge hybrid architecture to identify species from spectrograms, optimized for real-time biodiversity monitoring.

## 🚀 Project Achievements
* **Elite Precision:** Achieved **77.75% validation accuracy** in Epoch 10.
* **Computational & Thermal Efficiency:** Implemented **offline caching**, achieving a **12x speedup** in training (from ~2 hours to ~10 min per epoch).
* **PhD-Level Architecture:** Integration of**EfficientNet + Transformers** with advanced preprocessing (**PCEN + Spectral Gating**).

## 📊 Training History (Bioacoustics Model)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3.38 | 13.88% | 2.44 | 28.75% | Warm-up |
| 5 | 0.94 | 70.94% | 1.04 | 70.25% | **>70% Milestone** |
| 8 | 0.53 | 82.81% | 0.92 | 73.75% | Stable Model |
| 9 | 0.47 | 85.31% | 0.91 | 75.75% | Optimization |
| **10** | **0.41** | **87.75%** | **0.82** | **77.75%** | **FINAL MVP** |

## 🛠️ Repository Structure
* `src/`: Core scripts for training, data loading, and model architecture.
* `predict.py`: Inference script to classify new audio files.
* `requirements.txt`: Necessary dependencies to replicate the development environment.
* `checkpoints/`: (External Download) Contains the weights file `best_model.pth`.

## 💻 Usage & Inference
To verify the model with a test audio file, run:
```bash
python predict.py
```
Due to the parameter density (151 MB), the final model is hosted externally: 👉 (https://1drv.ms/f/c/E911E7181C90BA7E/IgB4rFVopgK3Q5BSsBa93h1wATPtDnp63mZBBAgcxtDfntI?e=g05NqD)

Developed by: BioPhys-Tech Lab 

