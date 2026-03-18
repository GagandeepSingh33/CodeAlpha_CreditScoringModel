import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

print("Step 1: Imports done")
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 120000, n),
    "loan_amount": np.random.randint(5000, 50000, n),
    "loan_tenure": np.random.randint(1, 10, n),
    "num_credit_cards": np.random.randint(1, 6, n),
    "existing_debts": np.random.randint(0, 30000, n),
    "missed_payments": np.random.randint(0, 10, n),
    "employment_type": np.random.choice(["Salaried","Self-Employed","Unemployed"], n, p=[0.6,0.3,0.1]),
    "education": np.random.choice(["High School","Graduate","Post-Graduate"], n),
})
score = (df["income"]/120000)*40 + (1 - df["missed_payments"]/10)*40 + (1 - df["existing_debts"]/30000)*20
df["creditworthy"] = (score > 55).astype(int)
print("Step 2: Dataset created")
print(df.head())
print(df["creditworthy"].value_counts())
df["debt_to_income"] = df["existing_debts"] / (df["income"] + 1)
df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
df["emi"] = df["loan_amount"] / (df["loan_tenure"] * 12)
le = LabelEncoder()
df["employment_type"] = le.fit_transform(df["employment_type"])
df["education"] = le.fit_transform(df["education"])
print("Step 3: Feature engineering done")
FEATURES = ["age","income","loan_amount","loan_tenure","num_credit_cards","existing_debts","missed_payments","employment_type","education","debt_to_income","loan_to_income","emi"]
X = df[FEATURES]
y = df["creditworthy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
print("Step 4: Split done - Train:", X_train_sc.shape, "Test:", X_test_sc.shape)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:,1]
    roc_auc = roc_auc_score(y_test, y_proba)
    results[name] = {"model":model,"y_pred":y_pred,"y_proba":y_proba,"roc_auc":roc_auc}
    print("\n" + "="*50)
    print("MODEL:", name)
    print(classification_report(y_test, y_pred, target_names=["Bad Credit","Good Credit"]))
    print("ROC-AUC:", round(roc_auc, 4))
fig, axes = plt.subplots(1, 3, figsize=(16,4))
for ax, (name, res) in zip(axes, results.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, res["y_pred"]), display_labels=["Bad","Good"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("Confusion matrices saved")
plt.figure(figsize=(8,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=name+" (AUC="+str(round(res["roc_auc"],3))+")")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()
print("ROC curves saved")
rf = results["Random Forest"]["model"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
imp.plot(kind="barh", color="steelblue", figsize=(8,6))
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Feature importance saved")
applicant = pd.DataFrame([{"age":35,"income":75000,"loan_amount":20000,"loan_tenure":5,"num_credit_cards":2,"existing_debts":5000,"missed_payments":1,"employment_type":1,"education":2,"debt_to_income":5000/75001,"loan_to_income":20000/75001,"emi":20000/60}])
pred = results["Random Forest"]["model"].predict(scaler.transform(applicant))[0]
prob = results["Random Forest"]["model"].predict_proba(scaler.transform(applicant))[0][1]
print("\nSINGLE APPLICANT PREDICTION")
print("Result:", "GOOD CREDIT" if pred==1 else "BAD CREDIT")
print("Confidence:", str(round(prob*100,1))+"%")
