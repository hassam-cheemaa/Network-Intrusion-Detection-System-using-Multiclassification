import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import lightgbm as lgb
import joblib
import os

# 1. SETUP
os.makedirs('model', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def save_plot(fig, filename):
    path = os.path.join('plots', filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated Plot: {path}")

# 2. LOAD & MERGE DATA
print("Loading Data...")
df1 = pd.read_csv('data/UNSW_NB15_testing-set.csv') 
df2 = pd.read_csv('data/UNSW_NB15_training-set.csv') 
full_df = pd.concat([df1, df2], ignore_index=True)
if 'id' in full_df.columns: full_df.drop('id', axis=1, inplace=True)

# 3. FEATURE ENGINEERING (The Accuracy Booster)
print("Engineering Features...")
# Create a 'rate' feature (Bytes per Second) - Highly predictive
full_df['bytes_per_sec'] = (full_df['sbytes'] + full_df['dbytes']) / (full_df['dur'] + 1e-5)

# 4. REPORTS: DATA DISTRIBUTION
print("Generating Reports...")
plt.figure(figsize=(10, 6))
order = full_df['attack_cat'].value_counts().index
sns.countplot(y=full_df['attack_cat'], order=order, palette='viridis')
plt.title('Distribution of Attack Categories')
save_plot(plt.gcf(), 'class_distribution_raw.png')

plt.figure(figsize=(10, 8))
numeric_df = full_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation')
save_plot(plt.gcf(), 'correlation_heatmap.png')

# 5. ENCODING
print("Encoding...")
# Frequency Encoding
cat_cols = ['proto', 'service', 'state']
for col in cat_cols:
    freq_map = full_df[col].value_counts(normalize=True).to_dict()
    full_df[col] = full_df[col].map(freq_map)
    joblib.dump(freq_map, f'model/freq_map_{col}.pkl')

# Target Encoding
target_le = LabelEncoder()
full_df['attack_cat'] = target_le.fit_transform(full_df['attack_cat'])
joblib.dump(target_le, 'model/target_encoder.pkl')

# 6. SPLIT & SCALE
X = full_df.drop(['label', 'attack_cat'], axis=1)
y = full_df['attack_cat']
joblib.dump(X.columns.tolist(), 'model/selected_features.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Log Transform (Helps with massive byte counts)
skewed = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'bytes_per_sec']
for col in skewed:
    if col in X_train.columns:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')

# 7. TRAIN MODEL (High Accuracy Mode)
print("Training LightGBM (Max Accuracy)...")
# Removed class_weight='balanced' to prioritize overall Accuracy score
model = lgb.LGBMClassifier(
    objective='multiclass', num_class=len(target_le.classes_),
    n_estimators=2000,      # High trees
    learning_rate=0.02,     # High precision
    num_leaves=60,          # Complex decisions
    random_state=42, n_jobs=-1, verbose=-1
)
model.fit(X_train, y_train)

# 8. EVALUATION
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🚀 FINAL ACCURACY: {acc*100:.2f}%")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 10))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_le.classes_)
disp.plot(cmap='Greens', ax=ax, xticks_rotation='vertical', values_format='d', colorbar=False)
save_plot(fig, 'confusion_matrix_ultra.png')

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
top_features = [X.columns[i] for i in indices]
plt.figure(figsize=(12, 6))
plt.bar(range(20), importances[indices], align="center", color='#00C853')
plt.xticks(range(20), top_features, rotation=45, ha='right')
plt.title("Top 20 Critical Features")
save_plot(plt.gcf(), 'feature_importance_lgbm.png')

joblib.dump(model, 'model/nids_xgboost_model.pkl')
print("✅ DONE! You can now run 'streamlit run app.py'")