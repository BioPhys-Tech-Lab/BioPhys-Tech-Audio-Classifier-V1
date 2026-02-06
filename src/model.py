import torch
import torch.nn as nn
import timm

class BioAcousticModel(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=True):
        super(BioAcousticModel, self).__init__()
        
        # --- 1. Backbone (Feature Extractor) ---
        # CNN robusta (EfficientNet) para extraer texturas del espectrograma
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        
        # Eliminar el clasificador original para obtener los features
        # EfficientNet features shape: [Batch, C, H, W]
        self.feature_dim = self.backbone.classifier.in_features
        self.backbone.reset_classifier(0, '') # Remove head
        
        # --- 2. Temporal Modeling (Transformer) ---
        # "Audio Spectrogram Transformer" concept: Treat time frames as tokens
        # Aplanamos H (freq) y W (time) into sequence
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        # --- 3. Attention Mechanism (Pooling) ---
        # "Attention Pooling" para que el modelo decida qué partes del audio son relevantes
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # --- 4. Classifier ---
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, 1, Freq, Time] (Spectrogram RGB-like but 1 channel if handled)
        # EfficientNet expects 3 channels usually, but we initialized with in_chans=1
        
        # 1. Extract Features
        # Output: [Batch, FeatureDim, H, W]
        features = self.backbone.forward_features(x)
        
        # 2. Reshape for Transformer
        # [Batch, FeatureDim, H, W] -> [Batch, FeatureDim, SeqLen] -> [Batch, SeqLen, FeatureDim]
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(b, h*w, c)
        
        # 3. Apply Transformer
        # Captura relaciones temporales largas (cantos complejos)
        attended_features = self.transformer(features)
        
        # 4. Attention Pooling
        # Weights: [Batch, SeqLen, 1]
        weights = self.attention(attended_features)
        
        # Weighted Sum: [Batch, FeatureDim]
        context_vector = torch.sum(weights * attended_features, dim=1)
        
        # 5. Classify
        # Add Dropout for regularization (PhD Level optimization for small datasets)
        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)
        
        return logits, weights

if __name__ == "__main__":
    # Sanity Check
    model = BioAcousticModel(num_classes=50)
    dummy_input = torch.randn(2, 1, 128, 300) # [Batch, 1, Freq_Mels, Time_Frames]
    output, attn = model(dummy_input)
    print(f"Model Output Shape: {output.shape}") # Should be [2, 50]
    print("PhD Architecture Initialized Successfully.")
