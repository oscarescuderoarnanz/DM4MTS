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

# Global parameters
num_epochs = 100
n_trials = 50
patience = 10
min_delta = 0.001
validation_ratio = 0.2
num_splits = 5
results_dir = "./Results"
os.makedirs(results_dir, exist_ok=True)

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# Column MLP
class ColumnMLP(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ColumnMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        batch_size, time_steps, features = x.shape
        embeddings = []
        for t in range(time_steps):
            feature_vector = x[:, t, :]
            embedding = self.mlp(feature_vector)
            embeddings.append(embedding.unsqueeze(1))
        column_embeddings = torch.cat(embeddings, dim=1)
        return column_embeddings

class RowMLP(nn.Module):
    def __init__(self, embedding_dim, reduced_time):
        super(RowMLP, self).__init__()
        self.reduced_time = reduced_time
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),  # Correct dimension order
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        # Add time reduction layer
        self.time_reducer = nn.Linear(embedding_dim, reduced_time) if embedding_dim != reduced_time else nn.Identity()
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, embedding_dim)
        batch_size, time_steps, embedding_dim = x.shape
        
        # Process each time step independently
        processed = []
        for t in range(time_steps):
            # Get features for this time step (batch_size, embedding_dim)
            time_slice = x[:, t, :]
            # Apply MLP
            transformed = self.mlp(time_slice)
            processed.append(transformed.unsqueeze(1))
        
        # Stack along time dimension (batch_size, time_steps, embedding_dim)
        x = torch.cat(processed, dim=1)
        
        # Reduce time dimension if needed
        if self.reduced_time != time_steps:
            # Transpose to (batch_size, embedding_dim, time_steps)
            x = x.transpose(1, 2)
            # Interpolate time dimension
            x = nn.functional.interpolate(x, size=self.reduced_time, mode='linear', align_corners=False)
            # Transpose back (batch_size, reduced_time, embedding_dim)
            x = x.transpose(1, 2)
        
        return x

# Final MLP
class FinalMLP(nn.Module):
    def __init__(self, embedding_dim, reduced_time, output_size):
        super(FinalMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * reduced_time, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.mlp(x)

class ColumnRowMLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, reduced_time, output_size):
        super(ColumnRowMLP, self).__init__()
        self.column_mlp = ColumnMLP(input_dim, embedding_dim)
        self.row_mlp = RowMLP(embedding_dim, reduced_time)
        self.final_mlp = FinalMLP(embedding_dim, reduced_time, output_size)

    def forward(self, x):
        # Input shape: (batch_size, time_steps, input_dim)
        column_embeddings = self.column_mlp(x)  # (batch_size, time_steps, embedding_dim)
        row_embeddings = self.row_mlp(column_embeddings)  # (batch_size, reduced_time, embedding_dim)
        output = self.final_mlp(row_embeddings)  # (batch_size, output_size)
        return output.squeeze(-1)  # Ensure output is (batch_size,)

# Objective function for Optuna
def objective(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 4, 128)
    # In your objective function:
    reduced_time = trial.suggest_int('reduced_time', 5, min(20, time_steps))  # Don't exceed original time steps
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    model = ColumnRowMLP(
        input_dim=features,
        embedding_dim=embedding_dim,
        reduced_time=reduced_time,
        output_size=1
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

            if early_stopping.should_stop(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final validation metrics
    probabilities = torch.sigmoid(val_outputs).numpy()
    predicted = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_val.numpy(), predicted)
    roc_auc = roc_auc_score(y_val.numpy(), probabilities)
    sensitivity = recall_score(y_val.numpy(), predicted)
    cm = confusion_matrix(y_val.numpy(), predicted)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("sensitivity", sensitivity)
    trial.set_user_attr("specificity", specificity)

    return val_loss

# Metrics dictionary
metrics = {'accuracy': [], 'roc_auc': [], 'sensitivity': [], 'specificity': []}

# Loop through each data split
for split in range(1, num_splits + 1):
    print(f"\nProcessing split {split}/{num_splits}")
    
    # Load data
    X = np.load(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/X_train_tensor.npy")
    y = pd.read_csv(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/y_train_tensor.csv")[['MR']].MR.values
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_ratio, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_val = torch.tensor(y_val).float()

    # Get dimensions
    time_steps = X_train.shape[1]
    features = X_train.shape[2]

    # Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Store results
    metrics['accuracy'].append(study.best_trial.user_attrs['accuracy'])
    metrics['roc_auc'].append(study.best_trial.user_attrs['roc_auc'])
    metrics['sensitivity'].append(study.best_trial.user_attrs['sensitivity'])
    metrics['specificity'].append(study.best_trial.user_attrs['specificity'])

    # Save results
    split_dir = os.path.join(results_dir, f"split_{split}")
    os.makedirs(split_dir, exist_ok=True)
    
    results = {
        "hyperparameters": study.best_params,
        "metrics": {
            "accuracy": study.best_trial.user_attrs['accuracy'],
            "roc_auc": study.best_trial.user_attrs['roc_auc'],
            "sensitivity": study.best_trial.user_attrs['sensitivity'],
            "specificity": study.best_trial.user_attrs['specificity']
        }
    }
    
    with open(os.path.join(split_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Split {split} saved at {split_dir}")

# Summary of metrics
print("\nSummary of All Splits:")
for metric in metrics:
    mean_value = np.mean(metrics[metric])
    std_value = np.std(metrics[metric])
    print(f"{metric.capitalize()} - Mean: {mean_value:.4f}, Std: {std_value:.4f}")

# Save summary to a JSON file
summary = {metric: {"mean": float(np.mean(values)), "std": float(np.std(values))} 
           for metric, values in metrics.items()}
with open(os.path.join(results_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\nSummary saved to ./Results/summary.json")