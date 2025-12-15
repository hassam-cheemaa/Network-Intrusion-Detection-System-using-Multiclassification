import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import joblib
import os

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# 2. LOAD MODEL ARTIFACTS
print("Loading model and encoders...")
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'nids_xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    target_le = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
    # Load frequency maps
    freq_map_proto = joblib.load(os.path.join(MODEL_DIR, 'freq_map_proto.pkl'))
    freq_map_service = joblib.load(os.path.join(MODEL_DIR, 'freq_map_service.pkl'))
    freq_map_state = joblib.load(os.path.join(MODEL_DIR, 'freq_map_state.pkl'))
except FileNotFoundError as e:
    print(f"Error: Missing file {e}. Did you run 'train_model_complete.py'?")
    exit()

# 3. LOAD & PREPARE DATA (Exact same steps as training)
print("Loading and preparing data...")
df1 = pd.read_csv('data/UNSW_NB15_testing-set.csv') 
df2 = pd.read_csv('data/UNSW_NB15_training-set.csv') 
full_df = pd.concat([df1, df2], ignore_index=True)

if 'id' in full_df.columns: full_df.drop('id', axis=1, inplace=True)

# Feature Engineering (Velocity)
full_df['bytes_per_sec'] = (full_df['sbytes'] + full_df['dbytes']) / (full_df['dur'] + 1e-5)

# Encoding (Using saved maps to be consistent)
full_df['proto'] = full_df['proto'].map(freq_map_proto)
full_df['service'] = full_df['service'].map(freq_map_service)
full_df['state'] = full_df['state'].map(freq_map_state)
full_df['attack_cat'] = target_le.transform(full_df['attack_cat'])

# 4. SPLIT (Must be random_state=42 to match your training split)
X = full_df.drop(['label', 'attack_cat'], axis=1)
y = full_df['attack_cat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. PREPROCESS TEST SET
# Log Transform
skewed = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'bytes_per_sec']
for col in skewed:
    if col in X_test.columns:
        X_test[col] = np.log1p(X_test[col])

# Scale (Using saved scaler)
X_test_scaled = scaler.transform(X_test)

# 6. CALCULATE ACCURACY
print("Running predictions...")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"✅ MODEL ACCURACY: {acc*100:.2f}%")
print("-" * 30)

# Optional: detailed report
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))