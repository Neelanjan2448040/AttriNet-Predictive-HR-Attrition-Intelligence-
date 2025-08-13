# train.py
import pandas as pd
import numpy as np
import joblib  # For saving models and objects
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

print("--- Starting Model Training Pipeline ---")

# 1. Load Data
print("Step 1/5: Loading dataset...")
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Preprocess Data
print("Step 2/5: Preprocessing data...")
# Drop unused constant cols if present
to_drop = [c for c in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'] if c in df.columns]
df_proc = df.drop(columns=to_drop)

# Label-encode categorical columns and remember encoders
label_encoders = {}
for c in df_proc.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_proc[c] = le.fit_transform(df_proc[c])
    label_encoders[c] = le

# Fill na if any (simple)
df_proc = df_proc.fillna(0)

# Features & target
X = df_proc.drop(columns=['Attrition'])
y = df_proc['Attrition']

# Standardize numeric features and save the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
X_scaled = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns, index=X.index)

# Merge scaled numeric with encoded categoricals
X_prepared = X.copy()
X_prepared[X_scaled.columns] = X_scaled
print("Data preprocessing complete.")

# 3. Train Models
print("Step 3/5: Training models...")
# Train ANN (MLPClassifier)
print("  - Training ANN (MLP)...")
ann = MLPClassifier(
    hidden_layer_sizes=(64,32),
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
ann.fit(X_prepared, y)
print("  - ANN training complete.")

# Train TabNet
print("  - Training TabNet...")
tabnet = TabNetClassifier(
    seed=42,
    verbose=0,
    n_d=32, n_a=32,
    n_steps=3,
    gamma=1.3,
    lambda_sparse=1e-3
)
X_train_np = X_prepared.values.astype(np.float32)
y_train_np = y.values.astype(np.int64)
tabnet.fit(
    X_train_np, y_train_np,
    max_epochs=50,
    patience=10,
    batch_size=min(256, len(X_prepared)//4),
    virtual_batch_size=min(64, len(X_prepared)//8),
    eval_set=None
)
print("  - TabNet training complete.")


# Train RandomForest (for fallback and comparison)
print("  - Training RandomForest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_prepared, y)
print("  - RandomForest training complete.")
print("All models trained successfully!")


# 4. Bundle artifacts for saving
print("Step 4/5: Bundling artifacts...")
artifacts = {
    'ann_model': ann,
    'tabnet_model': tabnet,
    'rf_model': rf,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'X_prepared_columns': X_prepared.columns.tolist() # Save column order
}
print("Artifacts bundled.")

# 5. Save artifacts to a file
print("Step 5/5: Saving artifacts to 'model_artifacts.joblib'...")
joblib.dump(artifacts, 'model_artifacts.joblib')

print("--- Model Training Pipeline Finished Successfully! ---")
print("You can now run the Streamlit app.")