from pathlib import Path
import pandas as pd
import numpy as np
from model import train_and_save, load_model
from feature_engineering import FEATURE_ORDER

# 1. Load Data
TRAINING_DATA = "training_data.csv"
MODEL_PATH = "credit_model.pkl"

if not Path(TRAINING_DATA).exists():
    print(f"Error: {TRAINING_DATA} not found. Run generate_compatible_data.py first.")
    exit(1)

df = pd.read_csv(TRAINING_DATA)
print(f"Loaded {len(df)} rows from {TRAINING_DATA}")

# 2. Prepare Features and Labels
X = df[FEATURE_ORDER].values.astype(np.float64)
y = df["defaulted"].values

# 3. Train Model
print("Training model...")
model, accuracy, path = train_and_save(X, y, model_path=MODEL_PATH)
print(f"Model trained with accuracy: {accuracy:.4f}")
print(f"Model saved to {path}")
