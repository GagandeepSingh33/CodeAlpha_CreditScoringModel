# ============================================================
# TRAIN AND SAVE MODEL
# ============================================================
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# ── 1. GENERATE DATASET & SAVE TO data/ FOLDER ──────────────
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "age":              np.random.randint(21, 65, n),
    "income":           np.random.randint(20000, 120000, n),
    "loan_amount":      np.random.randint(5000, 50000, n),
    "loan_tenure":      np.random.randint(1, 10, n),
    "num_credit_cards": np.random.randint(1, 6, n),
    "existing_debts":   np.random.randint(0, 30000, n),
    "missed_payments":  np.random.randint(0, 10, n),
    "employment_type":  np.random.choice(["Salaried","Self-Employed","Unemployed"], n, p=[0.6,0.3,0.1]),
    "education":        np.random.choice(["High School","Graduate","Post-Graduate"], n),
})

score = (
    (df["income"] / 120000) * 40
  + (1 - df["missed_payments"] / 10) * 40
  + (1 - df["existing_debts"] / 30000) * 20
)
df["creditworthy"] = (score > 55).astype(int)

# Save dataset to data/ folder
os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_data.csv", index=False)
print("Dataset saved to data/credit_data.csv")
print(df.head())

# ── 2. FEATURE ENGINEERING ───────────────────────────────────
df["debt_to_income"] = df["existing_debts"] / (df["income"] + 1)
df["loan_to_income"] = df["loan_amount"]    / (df["income"] + 1)
df["emi"]            = df["loan_amount"]    / (df["loan_tenure"] * 12)

le = LabelEncoder()
df["employment_type"] = le.fit_transform(df["employment_type"])
df["education"]       = le.fit_transform(df["education"])

# ── 3. TRAIN/TEST SPLIT ──────────────────────────────────────
FEATURES = [
    "age", "income", "loan_amount", "loan_tenure",
    "num_credit_cards", "existing_debts", "missed_payments",
    "employment_type", "education",
    "debt_to_income", "loan_to_income", "emi"
]

X = df[FEATURES]
y = df["creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. TRAIN MODEL ───────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sc, y_train)

y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]

print("\nModel Performance:")
print(classification_report(y_test, y_pred, target_names=["Bad Credit","Good Credit"]))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

# ── 5. SAVE MODEL & SCALER ───────────────────────────────────
os.makedirs("models", exist_ok=True)
pickle.dump(model,  open("models/credit_model.pkl",  "wb"))
pickle.dump(scaler, open("models/scaler.pkl",        "wb"))
print("\nModel saved to models/credit_model.pkl")
print("Scaler saved to models/scaler.pkl")
print("\nTraining complete!")