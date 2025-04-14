# Import required libraries with warnings suppressed
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
import optuna
import numpy as np
import pandas as pd
import os
import json
import math

# Global configuration parameters
num_epochs = 100
n_trials = 50
patience = 10             # Early stopping patience
min_delta = 0.001         # Minimum improvement for early stopping
validation_ratio = 0.2    # Ratio for validation split
num_splits = 5            # Number of data splits for cross-validation
results_dir = "./Results"  # Directory to save results
os.makedirs(results_dir, exist_ok=True)  # Create directory if not exists

class PositionalEncoding(nn.Module):
    """Implements positional encoding for Transformer models"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.register_buffer('pe', pe)  # Register as buffer to avoid being a learnable parameter
        
    def forward(self, x):
        """Add positional encoding to input tensor"""
        x = x + self.pe[:x.size(1)]  # Add encoding for each position
        return self.dropout(x)  # Apply dropout

class TransformerProcessor(nn.Module):
    """Transformer-based temporal processor"""
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerProcessor, self).__init__()
        
        # Validate that embedding dimension is divisible by number of attention heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Initialize components
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)  # Final classification layer

    def forward(self, embeddings):
        """Forward pass through the transformer"""
        embeddings = self.pos_encoder(embeddings)  # Add positional encoding
        transformer_out = self.transformer_encoder(embeddings)  # Process through transformer
        output = self.fc(transformer_out[:, -1, :])  # Take last timestep for classification
        return output

class TemporalProcessingModel(nn.Module):
    """Complete temporal processing model with embedding and transformer"""
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size, model_type="Transformer", 
                 num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalProcessingModel, self).__init__()
        self.model_type = model_type
        
        # Temporal embedding layer (MLP)
        self.temporal_embedding = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Temporal processor (Transformer)
        if model_type == "Transformer":
            self.temporal_processor = TransformerProcessor(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError("model_type must be 'Transformer'")

    def forward(self, x):
        """Forward pass through the complete model"""
        # Apply temporal embedding to each timestep
        batch_size, time_steps, features = x.shape
        x = x.view(-1, features)  # Flatten for embedding
        embeddings = self.temporal_embedding(x)
        embeddings = embeddings.view(batch_size, time_steps, -1)  # Reshape back
        
        # Process through transformer
        output = self.temporal_processor(embeddings)
        return output.squeeze(-1)  # Remove last dimension for BCEWithLogitsLoss

class EarlyStopping:
    """Implements early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, current_loss):
        """Check if training should stop early"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    embedding_dim = trial.suggest_int('embedding_dim', 4, 128) 
    hidden_dim = trial.suggest_int('hidden_dim', 4, 128)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    # Transformer-specific parameters
    possible_heads = [h for h in [2, 4, 6, 8] if embedding_dim % h == 0]  # Ensure divisibility
    num_heads = trial.suggest_categorical('num_heads', possible_heads)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)

    # Initialize model
    model = TemporalProcessingModel(
        input_size=features,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_size=1,
        model_type="Transformer",
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

            if early_stopping.should_stop(val_loss):
                break

    # Calculate metrics
    probabilities = torch.sigmoid(val_outputs).numpy()
    predicted = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_val.numpy(), predicted)
    roc_auc = roc_auc_score(y_val.numpy(), probabilities)
    sensitivity = recall_score(y_val.numpy(), predicted)
    cm = confusion_matrix(y_val.numpy(), predicted)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Store trial information
    trial.set_user_attr("model", model)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("sensitivity", sensitivity)
    trial.set_user_attr("specificity", specificity)

    return val_loss

# Initialize metric storage
metrics = {
    'accuracy': [], 
    'roc_auc': [], 
    'sensitivity': [], 
    'specificity': []
}

# Main training loop across splits
for split in range(1, num_splits + 1):
    print(f"\nProcessing split {split}...")
    
    # Load and prepare data
    X = np.load(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/X_train_tensor.npy")
    y = pd.read_csv(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/y_train_tensor.csv")[['MR']].MR.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_ratio, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_val = torch.tensor(y_val).float()

    features = X_train.shape[2]  # Get number of features

    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Save results
    best_trial = study.best_trial
    split_dir = os.path.join(results_dir, f"split_{split}")
    os.makedirs(split_dir, exist_ok=True)
    
    # Save hyperparameters and metrics
    results = {
        "hyperparameters": best_trial.params,
        "metrics": {metric: best_trial.user_attrs[metric] for metric in metrics}
    }
    
    with open(os.path.join(split_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save model weights
    torch.save(best_trial.user_attrs["model"].state_dict(), os.path.join(split_dir, "model.pt"))

    # Store metrics for final summary
    for metric in metrics:
        metrics[metric].append(best_trial.user_attrs[metric])

    print(f"Completed split {split}")

# Generate and save final summary
summary = {
    metric: {
        "mean": float(np.mean(values)),
        "std": float(np.std(values))
    } for metric, values in metrics.items()
}
with open(os.path.join(results_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

# Print final metrics
print("\nFinal metrics summary:")
for metric, stats in summary.items():
    print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")