# ============================================================
# FLASK WEB APP - With Graphs
# ============================================================
from flask import Flask, render_template, request
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly
import json

app = Flask(__name__)

model  = pickle.load(open("models/credit_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl",       "rb"))

FEATURES = [
    "age", "income", "loan_amount", "loan_tenure",
    "num_credit_cards", "existing_debts", "missed_payments",
    "employment_type", "education",
    "debt_to_income", "loan_to_income", "emi"
]

FEATURE_LABELS = [
    "Age", "Income", "Loan Amount", "Loan Tenure",
    "Credit Cards", "Existing Debts", "Missed Payments",
    "Employment Type", "Education",
    "Debt-to-Income", "Loan-to-Income", "EMI"
]

def make_gauge(confidence, result):
    color = "#27ae60" if result == "GOOD CREDIT" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Credit Score Confidence", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#fadbd8"},
                {"range": [40, 70], "color": "#fdebd0"},
                {"range": [70,100], "color": "#d5f5e3"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.75,
                "value": confidence
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30),
                      paper_bgcolor="rgba(0,0,0,0)")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_feature_bar(features_raw):
    rf = model
    importances = rf.feature_importances_
    weighted = [abs(features_raw[i]) * importances[i] for i in range(len(FEATURE_LABELS))]
    total = sum(weighted) or 1
    contributions = [round((w / total) * 100, 1) for w in weighted]

    colors = []
    for i, label in enumerate(FEATURE_LABELS):
        if label in ["Missed Payments", "Existing Debts", "Debt-to-Income", "Loan-to-Income"]:
            colors.append("#e74c3c")
        else:
            colors.append("#27ae60")

    fig = go.Figure(go.Bar(
        x=contributions,
        y=FEATURE_LABELS,
        orientation="h",
        marker_color=colors,
        text=[str(c)+"%" for c in contributions],
        textposition="outside"
    ))
    fig.update_layout(
        title="Feature Contribution to Decision",
        height=400,
        margin=dict(t=50, b=20, l=120, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(tickfont=dict(size=12))
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_risk_pie(features_raw):
    missed   = features_raw[6]
    debts    = features_raw[5] / 30000
    dti      = features_raw[9]
    income_r = 1 - (features_raw[1] / 120000)
    lti      = features_raw[10]

    values = [
        max(missed / 10, 0.01),
        max(debts,       0.01),
        max(dti,         0.01),
        max(income_r,    0.01),
        max(lti,         0.01)
    ]
    labels = ["Missed Payments", "Existing Debts",
              "Debt-to-Income",  "Income Risk", "Loan-to-Income"]
    colors = ["#e74c3c","#e67e22","#f1c40f","#3498db","#9b59b6"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent"
    ))
    fig.update_layout(
        title="Risk Factor Breakdown",
        height=350,
        margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age              = int(request.form["age"])
        income           = int(request.form["income"])
        loan_amount      = int(request.form["loan_amount"])
        loan_tenure      = int(request.form["loan_tenure"])
        num_credit_cards = int(request.form["num_credit_cards"])
        existing_debts   = int(request.form["existing_debts"])
        missed_payments  = int(request.form["missed_payments"])
        employment_type  = int(request.form["employment_type"])
        education        = int(request.form["education"])

        debt_to_income = existing_debts / (income + 1)
        loan_to_income = loan_amount    / (income + 1)
        emi            = loan_amount    / (loan_tenure * 12)

        features_raw = [
            age, income, loan_amount, loan_tenure,
            num_credit_cards, existing_debts, missed_payments,
            employment_type, education,
            debt_to_income, loan_to_income, emi
        ]

        features_sc = scaler.transform([features_raw])
        prediction  = model.predict(features_sc)[0]
        confidence  = round(model.predict_proba(features_sc)[0][1] * 100, 1)
        result      = "GOOD CREDIT" if prediction == 1 else "BAD CREDIT"
        color       = "green" if prediction == 1 else "red"

        gauge_json   = make_gauge(confidence, result)
        bar_json     = make_feature_bar(features_raw)
        pie_json     = make_risk_pie(features_raw)

        return render_template("index.html",
                               prediction=result,
                               confidence=confidence,
                               color=color,
                               gauge_json=gauge_json,
                               bar_json=bar_json,
                               pie_json=pie_json)
    except Exception as e:
        return render_template("index.html",
                               prediction="Error: " + str(e),
                               color="orange")

if __name__ == "__main__":
    app.run(debug=True)