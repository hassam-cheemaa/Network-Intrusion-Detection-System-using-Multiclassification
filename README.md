# 🛡️ Network Intrusion Detection System (NIDS)

An end-to-end, multi-class Network Intrusion Detection System built with Streamlit for interactive analysis and LightGBM for classification. The project uses the UNSW-NB15 dataset and provides both manual and batch analysis flows with heuristic overrides.

## 🚀 Features
- Interactive dashboard and analysis UI in app.py with manual inspection and batch upload tabs.
- Model training pipeline in train_model_complete.py (feature engineering, encoding, scaling, LightGBM training, plotting).
- Accuracy check and evaluation script in checkacc.py
- Preprocessed model artifacts persisted to `model/` (encoders, scaler, trained model).
- Plots and reports saved to `plots/`.
- Uses frequency encoding for categorical features and a `bytes_per_sec` velocity feature for improved accuracy.

## 🗂 Repository Structure
- [app.py](app.py) — Streamlit UI for dashboard and analysis.
- [train_model_complete.py](train_model_complete.py) — end-to-end training and report generation.
- [checkacc.py](checkacc.py) — reloads artifacts and prints accuracy/classification report.
- [data/UNSW_NB15_training-set.csv](data/UNSW_NB15_training-set.csv) — training data.
- [data/UNSW_NB15_testing-set.csv](data/UNSW_NB15_testing-set.csv) — testing data.
- `model/` — saved encoder maps, scaler, and trained model (`nids_xgboost_model.pkl`).
- `plots/` — generated plots (distribution, correlation, confusion matrix, feature importance).

## 🧠 Model & Features
Key pipeline steps (see [train_model_complete.py](train_model_complete.py)):
- Derived feature `bytes_per_sec = (sbytes + dbytes) / (dur + 1e-5)`.
- Frequency encoding for `proto`, `service`, `state`.
- Target encoding via `LabelEncoder`.
- Log transform on skewed numeric fields (`dur`, `sbytes`, `dbytes`, `sload`, `dload`, `bytes_per_sec`).
- Train/test split with stratification, MinMax scaling, LightGBM classifier.
- Artifacts saved to `model/` for reuse in the UI and evaluation.

## 🖥 Running the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure model artifacts exist (run training first if missing).
3. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Use the Dashboard for KPIs and charts; switch to **Analysis Engine** for manual inputs or batch CSV upload.

## 🏋️ Training the Model
From the project root:
```bash
python train_model_complete.py
```
This will:
- Load and merge the UNSW-NB15 train/test CSVs.
- Engineer features, encode, scale, and train LightGBM.
- Save artifacts to `model/` and plots to `plots/`.

## ✅ Evaluating Accuracy
After training, verify accuracy and classification report:
```bash
python checkacc.py
```

## 📦 Required Artifacts
- `model/nids_xgboost_model.pkl`
- `model/scaler.pkl`
- `model/target_encoder.pkl`
- `model/freq_map_proto.pkl`, `model/freq_map_service.pkl`, `model/freq_map_state.pkl`
- `model/selected_features.pkl`

## 📊 Data
The project expects the UNSW-NB15 CSVs:
- [data/UNSW_NB15_training-set.csv](data/UNSW_NB15_training-set.csv)
- [data/UNSW_NB15_testing-set.csv](data/UNSW_NB15_testing-set.csv)

Ensure they remain in `data/` with headers intact (drop `id` column handled in code).

## 📝 Notes
- Manual override heuristics in the UI flag certain traffic patterns before ML prediction (see [app.py](app.py)).
- Batch upload accepts CSV with the same schema as training/testing data.
- Plots (distribution, correlation, confusion matrix, feature importance) are auto-generated to `plots/`.
 
## Screenshot
<img width="1906" height="904" alt="image" src="https://github.com/user-attachments/assets/470953dc-78b2-47b2-84e7-0f695d98bbec" />
<img width="1897" height="876" alt="image" src="https://github.com/user-attachments/assets/90d091c2-0f01-4245-a177-79270c2f7a36" />
<img width="1265" height="649" alt="image" src="https://github.com/user-attachments/assets/396011f1-704c-4f1b-9371-49d2ca131b30" />
<img width="1377" height="781" alt="image" src="https://github.com/user-attachments/assets/53a5870a-fd4e-46ed-ae93-4ee03e21f04f" />

