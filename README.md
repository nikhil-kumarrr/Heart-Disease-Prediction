# Heart Disease Risk Predictor
An ML-powered heart disease risk assessment app built using Random Forest, Logistic Regression, and XGBoost with an interactive Streamlit dashboard.
Enter patient clinical data and instantly get an AI-powered cardiac risk prediction with confidence scores.

## Features
- ML-based heart disease risk prediction
- 3 models compared (RF, LR, XGBoost)
- Instant risk assessment with probability scores
- Clean hospital-themed UI
- Uses UCI Heart Disease dataset
- Real-time prediction engine
- Built-in clinical field reference guide

## How It Works

The system uses:

### 1️⃣ Dataset
UCI Heart Disease Dataset containing:
- Age, Sex, Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Max Heart Rate
- ST Depression (Oldpeak)
- Thalassemia, Major Vessels
- Target (Disease / No Disease)

### 2️⃣ Data Processing (Notebook)
- Missing value and duplicate check
- Class imbalance handling (RandomOverSampler)
- Feature scaling (StandardScaler)
- Train-test split (80/20, stratified)

### 3️⃣ ML Model
- 3 Models Trained → Logistic Regression, Random Forest, XGBoost
- Evaluation → Accuracy, Precision, Recall, F1, ROC-AUC
- Best Model → Random Forest (ROC-AUC: 91%)
- Stored as → best_model.pkl + scaler.pkl

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Streamlit
- Joblib

## Installation & Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
```

### 3️⃣ Activate environment

#### Windows
```bash
venv\Scripts\activate
```

#### Mac/Linux
```bash
source venv/bin/activate
```

### 4️⃣ Install requirements
```bash
pip install -r requirements.txt
```

### 5️⃣ Run Streamlit app
```bash
streamlit run app.py
```

## Project Structure
```
│── app.py
│── best_model.pkl
│── scaler.pkl
│── Heart_disease_prediction.ipynb
│── requirements.txt
└── README.md
```

## Model Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | 80.3% | 91.4% |
| Logistic Regression | 78.7% | 86.5% |
| XGBoost | 78.7% | 84.3% |

## Dataset
Available on Kaggle: https://www.kaggle.com/datasets/abhishek14398/heart-disease-classification

## Live Demo
https://ai-heartdisease-predictor.streamlit.app/

## Screenshots
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-02-27%20100131.png?raw=true)
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-02-27%20100145.png?raw=true)
