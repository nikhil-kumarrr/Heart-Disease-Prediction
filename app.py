import streamlit as st
import numpy as np
import joblib

model  = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #EEF4FB !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] { background: #0D2B45 !important; }
[data-testid="stSidebar"] * { color: #B0C4D8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important; font-family: 'Playfair Display', serif;
}
[data-testid="stSidebar"] label {
    color: #7FA8C9 !important; font-size: 0.76rem !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.07em;
}
.header-box {
    background: linear-gradient(135deg, #0D2B45 0%, #1565C0 100%);
    border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
}
.header-box h1 {
    color: #fff; font-family: 'Playfair Display', serif;
    font-size: 1.9rem; margin: 0 0 4px 0;
}
.header-box p { color: #90CAF9; margin: 0; font-size: 0.9rem; }

.info-box {
    background: #FFFFFF; border-radius: 12px; padding: 14px 16px;
    border-left: 4px solid #1565C0; margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.08);
}
.info-box .field-name {
    font-size: 0.8rem; font-weight: 700; color: #1565C0;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;
}
.info-box .field-desc { font-size: 0.82rem; color: #455A64; line-height: 1.5; }

.result-high {
    background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
    border: 2px solid #EF9A9A; border-radius: 14px;
    padding: 28px; text-align: center; margin-bottom: 20px;
}
.result-low {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border: 2px solid #A5D6A7; border-radius: 14px;
    padding: 28px; text-align: center; margin-bottom: 20px;
}
.result-high h2 { color: #B71C1C; font-family: 'Playfair Display', serif; font-size: 1.6rem; margin: 8px 0 6px; }
.result-low  h2 { color: #1B5E20; font-family: 'Playfair Display', serif; font-size: 1.6rem; margin: 8px 0 6px; }
.result-high p { color: #C62828; margin: 0; font-size: 0.95rem; }
.result-low  p { color: #2E7D32; margin: 0; font-size: 0.95rem; }

.prob-label {
    font-size: 0.78rem; font-weight: 600; color: #546E7A;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 5px; margin-top: 14px;
}
.bar-track { background: #E3EDF9; border-radius: 99px; height: 9px; overflow: hidden; margin-bottom: 4px; }
.bar-fill   { height: 100%; border-radius: 99px; }
.bar-val    { font-size: 0.82rem; font-weight: 700; margin-bottom: 12px; }

.section-title {
    font-size: 0.72rem; font-weight: 700; color: #78909C;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 20px 0 10px 0; border-bottom: 1px solid #CFD8DC; padding-bottom: 6px;
}

.summary-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
.chip {
    background: #FFFFFF; border: 1px solid #DDE6F0;
    border-radius: 11px; padding: 12px 16px;
    flex: 1; min-width: 100px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.06);
}
.chip .cv {
    font-size: 1.25rem; font-weight: 700;
    color: #0D47A1; font-family: 'Playfair Display', serif;
}
.chip .ck {
    font-size: 0.7rem; color: #78909C;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 3px;
}

.disclaimer {
    background: #FFF8E1; border-left: 4px solid #F9A825;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 0.8rem; color: #6D4C00; margin-top: 18px; line-height: 1.6;
}
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #0D2B45) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 13px 0 !important; font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important; box-shadow: 0 4px 14px rgba(21,101,192,0.3) !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>Heart Disease Risk Predictor</h1>
    <p>AI-powered clinical screening tool &nbsp;·&nbsp; For educational use only</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.6, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Form + Result
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    age = c1.slider("Age (years)", 20, 80, 50)
    sex = c2.selectbox("Sex", ["Male", "Female"])

    st.markdown('<div class="section-title">Clinical Measurements</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    resting_bp  = c3.number_input("Resting BP (mm Hg)", 80, 220, 130)
    cholestoral = c4.number_input("Cholesterol (mg/dl)", 100, 600, 240)

    c5, c6 = st.columns(2)
    max_hr  = c5.slider("Max Heart Rate", 60, 210, 150)
    oldpeak = c6.slider("Oldpeak (ST Depression)", 0.0, 6.5, 1.0, step=0.1)

    c7, c8 = st.columns(2)
    fasting_bs = c7.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
    exang      = c8.selectbox("Exercise-Induced Angina", ["No", "Yes"])

    st.markdown('<div class="section-title">Test Results</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    chest_pain = c9.selectbox("Chest Pain Type", [0, 1, 2, 3])
    restecg    = c10.selectbox("Resting ECG", [0, 1, 2])

    c11, c12, c13 = st.columns(3)
    slope       = c11.selectbox("ST Slope", [0, 1, 2])
    num_vessels = c12.selectbox("Major Vessels", [0, 1, 2, 3])
    thal        = c13.selectbox("Thal", [0, 1, 2, 3])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Risk Assessment")

    # ── Result ────────────────────────────────────────────────────────────────
    if predict_btn:
        input_data = np.array([[
            age,
            1 if sex == "Male" else 0,
            chest_pain, resting_bp, cholestoral,
            1 if fasting_bs == "Yes" else 0,
            restecg, max_hr,
            1 if exang == "Yes" else 0,
            oldpeak, slope, num_vessels, thal
        ]])

        input_scaled    = scaler.transform(input_data)
        prediction      = model.predict(input_scaled)[0]
        probability     = model.predict_proba(input_scaled)[0]
        prob_disease    = round(probability[1] * 100, 1)
        prob_no_disease = round(probability[0] * 100, 1)

        st.markdown("<br>", unsafe_allow_html=True)

        # Result card
        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
                <div style="font-size:2.5rem"></div>
                <h2>High Risk Detected</h2>
                <p>Model predicts <strong>{prob_disease}%</strong> probability of heart disease.</p>
                <div class="prob-label" style="text-align:left">Disease Probability</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{prob_disease}%;background:linear-gradient(90deg,#E53935,#B71C1C);"></div>
                </div>
                <div class="bar-val" style="color:#C62828;text-align:left">{prob_disease}%</div>
                <div class="prob-label" style="text-align:left">No Disease Probability</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{prob_no_disease}%;background:linear-gradient(90deg,#43A047,#1B5E20);"></div>
                </div>
                <div class="bar-val" style="color:#2E7D32;text-align:left">{prob_no_disease}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div style="font-size:2.5rem"></div>
                <h2>Low Risk</h2>
                <p>Model predicts <strong>{prob_no_disease}%</strong> probability of no heart disease.</p>
                <div class="prob-label" style="text-align:left">No Disease Probability</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{prob_no_disease}%;background:linear-gradient(90deg,#43A047,#1B5E20);"></div>
                </div>
                <div class="bar-val" style="color:#2E7D32;text-align:left">{prob_no_disease}%</div>
                <div class="prob-label" style="text-align:left">Disease Probability</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{prob_disease}%;background:linear-gradient(90deg,#E53935,#B71C1C);"></div>
                </div>
                <div class="bar-val" style="color:#C62828;text-align:left">{prob_disease}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Patient Summary chips
        st.markdown('<div class="section-title">Patient Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-grid">
            <div class="chip"><div class="cv">{age}</div><div class="ck">Age (yrs)</div></div>
            <div class="chip"><div class="cv">{"M" if sex == "Male" else "F"}</div><div class="ck">Sex</div></div>
            <div class="chip"><div class="cv">{max_hr}</div><div class="ck">Max HR</div></div>
            <div class="chip"><div class="cv">{resting_bp}</div><div class="ck">BP mmHg</div></div>
            <div class="chip"><div class="cv">{cholestoral}</div><div class="ck">Cholesterol</div></div>
            <div class="chip"><div class="cv">{oldpeak}</div><div class="ck">Oldpeak</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> This tool is intended for educational and research
            purposes only. It is not a substitute for professional clinical diagnosis. Always consult
            a qualified cardiologist or medical professional for health decisions.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Field Reference Guide
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="section-title">Field Reference Guide</div>', unsafe_allow_html=True)

    fields = [
        ("Age",
         "Patient's age in years. Risk of heart disease increases significantly with age, especially after 45 in men and 55 in women."),
        ("Sex",
         "Male = 1 · Female = 0. Men have a higher risk of heart disease compared to pre-menopausal women."),
        ("Chest Pain Type",
         "0 = Typical Angina (chest pain triggered by exertion) · 1 = Atypical Angina · 2 = Non-Anginal Pain (not heart-related) · 3 = Asymptomatic (no pain but disease may still be present)"),
        ("Resting BP",
         "Blood pressure measured at rest (mm Hg). Normal is around 120/80. Values above 140 indicate hypertension and increase cardiac risk."),
        ("Cholesterol",
         "Serum cholesterol level (mg/dl). Below 200 is normal, 240+ is high. Elevated cholesterol leads to arterial blockage over time."),
        ("Fasting Blood Sugar",
         "Is fasting blood sugar greater than 120 mg/dl? Yes = 1. An indicator of diabetes, which significantly raises heart disease risk."),
        ("Resting ECG",
         "0 = Normal · 1 = ST-T wave abnormality (irregular electrical activity) · 2 = Left Ventricular Hypertrophy (thickened heart wall)"),
        ("Max Heart Rate",
         "Maximum heart rate achieved during exercise. A lower peak may indicate reduced cardiac output and higher disease risk."),
        ("Exercise-Induced Angina",
         "Chest pain during physical exertion? Yes = strong risk indicator. Suggests inadequate blood supply to the heart during activity."),
        ("Oldpeak (ST Depression)",
         "ST segment depression on ECG post-exercise. 0 = normal. Higher values reflect greater cardiac stress and ischemia."),
        ("ST Slope",
         "0 = Upsloping (healthy) · 1 = Flat (moderate risk) · 2 = Downsloping (high risk — reduced myocardial blood flow)"),
        ("Major Vessels",
         "Number of major coronary vessels (0–3) with blockage on fluoroscopy. More blocked vessels = higher disease severity."),
        ("Thal",
         "0/1 = Normal · 2 = Fixed Defect (permanent myocardial damage) · 3 = Reversible Defect (temporary blood flow reduction)"),
    ]

    for name, desc in fields:
        st.markdown(f"""
        <div class="info-box">
            <div class="field-name">{name}</div>
            <div class="field-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)