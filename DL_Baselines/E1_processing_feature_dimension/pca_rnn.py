import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
import optuna
import numpy as np
import pandas as pd
import os
import json
import math

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

# Global parameters
n_trials = 50
num_epochs = 100       # Number of training epochs
patience = 10          # Early stopping patience
min_delta = 0.001      # Minimum improvement to be considered as progress
validation_ratio = 0.2 # Percentage of data used for validation
num_splits = 5         # Number of data splits
results_dir = "./Results"  # Directory to save results

# Create the results directory if it does not exist
os.makedirs(results_dir, exist_ok=True)

# Positional Encoding module for the Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create constant 'pe' matrix with values dependent on 
        # pos and i (dimension)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if d_model is odd, handle the last column separately
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer-based classifier definition
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(TransformerClassifier, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(d_model=hidden_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Fully connected layer for binary classification
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Project input to model dimension
        x = self.input_projection(x)  # shape: (batch_size, seq_len, hidden_size)
        x = self.pos_encoder(x)
        # Transformer expects input shape: (seq_len, batch_size, hidden_size)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        # Aggregate: mean pooling across the time dimension
        x = x.mean(dim=0)
        out = self.fc(x)
        return out

# Early stopping class definition
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    # Check if training should be stopped
    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    # Hiperparámetros a optimizar:
    n_components = trial.suggest_int('n_components', 4, 50) 
    # Restricción para que hidden_size sea múltiplo de 4:
    hidden_size = trial.suggest_int('hidden_size', 4, 128, step=4)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.3) 
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    # Dimensionality reduction using PCA
    X_reshaped = X_train.view(-1, X_train.shape[-1]).numpy()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_reshaped)
    X_reduced = torch.tensor(X_pca).view(X_train.shape[0], X_train.shape[1], n_components).float()

    # PCA transformation on validation data
    X_val_reshaped = X_val.view(-1, X_val.shape[-1]).numpy()
    X_val_pca = pca.transform(X_val_reshaped)
    X_val_reduced = torch.tensor(X_val_pca).view(X_val.shape[0], X_val.shape[1], n_components).float()

    # Initialize the Transformer model
    model = TransformerClassifier(input_size=n_components, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_reduced).squeeze(1)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loss calculation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_reduced).squeeze(1)
        val_loss = criterion(val_outputs, y_val).item()
        probabilities = torch.sigmoid(val_outputs).numpy()
        predicted = (probabilities >= 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_val.numpy(), predicted)
        roc_auc = roc_auc_score(y_val.numpy(), probabilities)
        sensitivity = recall_score(y_val.numpy(), predicted)
        cm = confusion_matrix(y_val.numpy(), predicted)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Store the model and PCA as user attributes of the trial
    trial.set_user_attr("model", model)
    trial.set_user_attr("pca", pca)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("sensitivity", sensitivity)
    trial.set_user_attr("specificity", specificity)

    # Return the validation loss as the objective value to minimize
    return val_loss


# Dictionary to store metrics for each split
metrics = {'accuracy': [], 'roc_auc': [], 'sensitivity': [], 'specificity': []}

# Loop through each data split
for split in range(1, num_splits + 1):
    # Load training data
    X = np.load(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/X_train_tensor.npy")
    y = pd.read_csv(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/y_train_tensor.csv")[['MR']].MR.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_ratio, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_val = torch.tensor(y_val).float()

    # Load test data
    X_test = np.load(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/X_test_tensor.npy")
    y_test = pd.read_csv(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/y_test_tensor.csv")[['MR']].MR.values
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    # Create and optimize the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Get the best trial and model
    best_trial = study.best_trial
    best_model = best_trial.user_attrs.get("model")
    best_pca = best_trial.user_attrs.get("pca")

    # Transform test data using the best PCA model
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_pca = best_pca.transform(X_test_reshaped)
    X_test_reduced = torch.tensor(X_test_pca).view(X_test.shape[0], X_test.shape[1], -1).float()

    # Evaluate the best model on test data
    best_model.eval()
    with torch.no_grad():
        test_outputs = best_model(X_test_reduced).squeeze(1)
        probabilities = torch.sigmoid(test_outputs).numpy()
        predicted = (probabilities >= 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted)
        roc_auc = roc_auc_score(y_test, probabilities)
        sensitivity = recall_score(y_test, predicted)
        cm = confusion_matrix(y_test, predicted)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Store metrics
    metrics['accuracy'].append(accuracy)
    metrics['roc_auc'].append(roc_auc)
    metrics['sensitivity'].append(sensitivity)
    metrics['specificity'].append(specificity)

    # Save results for the current split
    split_dir = os.path.join(results_dir, f"split_{split}")
    os.makedirs(split_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(split_dir, "best_model.ckpt"))

    # Save hyperparameters and metrics in a JSON file
    results = {
        "hyperparameters": study.best_params,
        "metrics": {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "sensitivity": sensitivity,
            "specificity": specificity
        },
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params
        }
    }
    with open(os.path.join(split_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSplit {split} saved at {split_dir}")

# Summary of metrics across all splits
print("\nSummary of All Splits:")
summary = {}
for metric in metrics:
    mean_value = np.mean(metrics[metric])
    std_value = np.std(metrics[metric])
    print(f"{metric.capitalize()} - Mean: {mean_value:.4f}, Std: {std_value:.4f}")
    summary[metric] = {"mean": mean_value, "std": std_value}

# Save summary to a JSON file
summary_path = os.path.join(results_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"\nSummary of all splits saved at {summary_path}")
