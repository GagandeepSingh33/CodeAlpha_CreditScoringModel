# Credit Scoring Model

A machine learning web application that predicts an individual's creditworthiness using past financial data. Built with Python, Scikit-learn, and Flask.

## Features
- Predicts credit risk as GOOD CREDIT or BAD CREDIT
- Trained on financial features including income, debts, missed payments
- Interactive web interface built with Flask
- Live charts powered by Plotly (Gauge, Pie, Bar charts)
- SMOTE used to handle class imbalance
- Model evaluated using Precision, Recall, F1-Score, ROC-AUC

## Tech Stack
- Language: Python 3.14
- ML Library: Scikit-learn
- Data Processing: Pandas, NumPy
- Imbalance Handling: Imbalanced-learn (SMOTE)
- Web Framework: Flask
- Visualization: Plotly
- Frontend: HTML, CSS

## ML Models Used
- Logistic Regression
- Decision Tree
- Random Forest (deployed - best performance)

## Installation

### 1. Clone the repository
git clone https://github.com/GagandeepSingh33/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Train the model
python train_model.py

### 5. Run the app
python app.py

### 6. Open in browser
http://127.0.0.1:5000

## Model Performance
| Model | F1-Score | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.84 | 0.91 |
| Decision Tree | 0.86 | 0.93 |
| Random Forest | 0.91 | 0.97 |

## Author
Gagandeep Singh
GitHub: https://github.com/GagandeepSingh33
