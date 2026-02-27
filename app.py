import streamlit as st
import numpy as np
import joblib

model  = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="🫀", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #F0F5FC !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1F35 0%, #0D2B45 100%) !important;
    border-right: 1px solid #1A3A55;
    min-width: 260px !important;
}
[data-testid="stSidebar"] * { color: #90AFC8 !important; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] .sidebar-logo {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] label {
    color: #5A8FAF !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #E0EAF4 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] p { color: #7BAAC9 !important; font-size: 0.82rem; }

/* ── Main Header ── */
.header-wrap {
    background: linear-gradient(135deg, #0A1F35 0%, #0D47A1 60%, #1565C0 100%);
    border-radius: 18px;
    padding: 36px 42px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(13,43,69,0.22);
    position: relative;
    overflow: hidden;
}
.header-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.header-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; right: 80px;
    width: 140px; height: 140px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.header-wrap h1 {
    font-family: 'Playfair Display', serif;
    color: #FFFFFF;
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.header-wrap p { color: #90CAF9; font-size: 0.92rem; font-weight: 400; }
.header-badges { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 0.76rem;
    color: #BBDEFB !important;
    font-weight: 500;
    backdrop-filter: blur(4px);
}
.pulse { display: inline-block; width: 8px; height: 8px; background: #4FC3F7; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(1.5)} }

/* ── Section Labels ── */
.section-label {
    font-size: 0.68rem; font-weight: 700; color: #90A4AE;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 22px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px; background: #CFD8DC;
}

/* ── Input Cards ── */
.input-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(13,43,69,0.07);
    border: 1px solid #DDE6F0;
}

/* ── Field Guide Cards ── */
.guide-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left: 4px solid #1565C0;
    box-shadow: 0 2px 8px rgba(21,101,192,0.07);
    transition: box-shadow 0.2s;
}
.guide-card:hover { box-shadow: 0 4px 16px rgba(21,101,192,0.15); }
.guide-card .fname {
    font-size: 0.72rem; font-weight: 700; color: #1565C0;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px;
}
.guide-card .fdesc { font-size: 0.82rem; color: #546E7A; line-height: 1.55; }

/* ── Result Cards ── */
.result-high {
    background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
    border: 1.5px solid #EF9A9A;
    border-radius: 16px; padding: 32px 28px; text-align: center;
    box-shadow: 0 4px 20px rgba(183,28,28,0.12);
    margin-top: 20px;
}
.result-low {
    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
    border: 1.5px solid #A5D6A7;
    border-radius: 16px; padding: 32px 28px; text-align: center;
    box-shadow: 0 4px 20px rgba(27,94,32,0.10);
    margin-top: 20px;
}
.result-high h2 { font-family:'Playfair Display',serif; color:#B71C1C; font-size:1.6rem; margin:10px 0 8px; }
.result-low  h2 { font-family:'Playfair Display',serif; color:#1B5E20; font-size:1.6rem; margin:10px 0 8px; }
.result-high p  { color:#C62828; font-size:0.9rem; }
.result-low  p  { color:#2E7D32; font-size:0.9rem; }
.result-icon    { font-size:3rem; line-height:1; }

/* ── Prob Bar ── */
.prob-wrap { margin-top: 20px; }
.prob-row  { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.prob-label { font-size:0.82rem; color:#546E7A; font-weight:500; }
.prob-val   { font-size:0.9rem;  font-weight:700; }
.bar-track  { background:#E3EDF9; border-radius:99px; height:9px; overflow:hidden; margin-bottom:14px; }
.bar-fill   { height:100%; border-radius:99px; }

/* ── Summary Chips ── */
.summary-grid { display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }
.chip {
    background:#F0F5FC; border:1px solid #DDE6F0;
    border-radius:10px; padding:10px 14px; flex:1; min-width:110px;
}
.chip .cv  { font-size:1.1rem; font-weight:700; color:#0D47A1; }
.chip .ck  { font-size:0.72rem; color:#78909C; text-transform:uppercase; letter-spacing:0.05em; margin-top:2px; }

/* ── Disclaimer ── */
.disclaimer {
    background:#FFF8E1; border-left:4px solid #FFB300;
    border-radius:0 10px 10px 0; padding:13px 16px;
    font-size:0.8rem; color:#6D4C00; margin-top:20px; line-height:1.6;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1565C0 0%, #0A1F35 100%) !important;
    color: white !important; border: none !important;
    border-radius: 11px !important; padding: 14px 0 !important;
    font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.02em !important;
    box-shadow: 0 5px 18px rgba(21,101,192,0.32) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 24px rgba(21,101,192,0.42) !important;
    transform: translateY(-2px) !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:1.6rem; margin-bottom:4px;">🫀</div>
        <div style="color:#FFFFFF !important; font-family:'Playfair Display',serif; font-size:1rem; font-weight:600;">CardioAI</div>
        <div style="color:#5A8FAF !important; font-size:0.75rem; margin-top:2px;">Clinical Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#3A6A8A !important;margin-bottom:12px;'>Patient Information</p>", unsafe_allow_html=True)

    age         = st.slider("Age (years)", 20, 80, 50)
    sex         = st.selectbox("Sex", ["Male", "Female"])
    resting_bp  = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
    cholestoral = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    max_hr      = st.slider("Max Heart Rate Achieved", 60, 210, 150)
    oldpeak     = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    fasting_bs  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    exang       = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    chest_pain  = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    restecg     = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    slope       = st.selectbox("ST Slope (0–2)", [0, 1, 2])
    num_vessels = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal        = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Run Risk Assessment")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <h1>🫀 Heart Disease Risk Predictor</h1>
    <p>Enter patient clinical data on the left panel to generate an AI-powered cardiac risk assessment.</p>
    <div class="header-badges">
        <span class="badge"><span class="pulse"></span>Live Prediction</span>
        <span class="badge">Random Forest Model</span>
        <span class="badge">ROC-AUC 91%</span>
        <span class="badge">303 Training Samples</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1.5, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Result + Summary
# ══════════════════════════════════════════════════════════════════════════════
with left:
    if not predict_btn:
        st.markdown("""
        <div class="input-card" style="text-align:center; padding:48px 24px;">
            <div style="font-size:3.5rem; margin-bottom:16px;">📋</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:#0D2B45; margin-bottom:8px;">
                Ready for Assessment
            </div>
            <div style="font-size:0.88rem; color:#78909C; max-width:320px; margin:0 auto; line-height:1.6;">
                Fill in the patient details in the left panel and click
                <strong>Run Risk Assessment</strong> to generate the prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="summary-grid">
            <div class="chip"><div class="cv">91%</div><div class="ck">ROC-AUC</div></div>
            <div class="chip"><div class="cv">80%</div><div class="ck">Accuracy</div></div>
            <div class="chip"><div class="cv">91%</div><div class="ck">Recall</div></div>
            <div class="chip"><div class="cv">77%</div><div class="ck">Precision</div></div>
            <div class="chip"><div class="cv">RF</div><div class="ck">Algorithm</div></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Predict ───────────────────────────────────────────────────────────
        input_data = np.array([[
            age,
            1 if sex == "Male" else 0,
            chest_pain, resting_bp, cholestoral,
            1 if fasting_bs == "Yes" else 0,
            restecg, max_hr,
            1 if exang == "Yes" else 0,
            oldpeak, slope, num_vessels, thal
        ]])

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        prob_disease    = round(probability[1] * 100, 1)
        prob_no_disease = round(probability[0] * 100, 1)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-icon">⚠️</div>
                <h2>High Risk — Disease Detected</h2>
                <p>The model predicts a <strong>{prob_disease}%</strong> probability of heart disease for this patient.</p>
                <div class="prob-wrap">
                    <div class="prob-row">
                        <span class="prob-label">🔴 Disease Probability</span>
                        <span class="prob-val" style="color:#C62828">{prob_disease}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{prob_disease}%;background:linear-gradient(90deg,#E53935,#B71C1C);"></div>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">🟢 No Disease Probability</span>
                        <span class="prob-val" style="color:#2E7D32">{prob_no_disease}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{prob_no_disease}%;background:linear-gradient(90deg,#43A047,#1B5E20);"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-icon">✅</div>
                <h2>Low Risk — No Disease Detected</h2>
                <p>The model predicts a <strong>{prob_no_disease}%</strong> probability of no heart disease for this patient.</p>
                <div class="prob-wrap">
                    <div class="prob-row">
                        <span class="prob-label">🟢 No Disease Probability</span>
                        <span class="prob-val" style="color:#2E7D32">{prob_no_disease}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{prob_no_disease}%;background:linear-gradient(90deg,#43A047,#1B5E20);"></div>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">🔴 Disease Probability</span>
                        <span class="prob-val" style="color:#C62828">{prob_disease}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{prob_disease}%;background:linear-gradient(90deg,#E53935,#B71C1C);"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Patient summary chips
        st.markdown('<div class="section-label">Patient Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-grid">
            <div class="chip"><div class="cv">{age}</div><div class="ck">Age (yrs)</div></div>
            <div class="chip"><div class="cv">{sex[0]}</div><div class="ck">Sex</div></div>
            <div class="chip"><div class="cv">{max_hr}</div><div class="ck">Max HR</div></div>
            <div class="chip"><div class="cv">{resting_bp}</div><div class="ck">BP mmHg</div></div>
            <div class="chip"><div class="cv">{cholestoral}</div><div class="ck">Cholesterol</div></div>
            <div class="chip"><div class="cv">{oldpeak}</div><div class="ck">Oldpeak</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            ⚕️ <strong>Medical Disclaimer:</strong> This tool is intended for educational and research
            purposes only. It is not a substitute for professional clinical diagnosis. Always consult
            a qualified cardiologist or medical professional for health decisions.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Field Reference Guide
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="section-label">📖 Clinical Field Reference</div>', unsafe_allow_html=True)

    fields = [
        ("Age",
         "Patient's age in years. Risk of heart disease increases significantly with age, especially after 45 in men and 55 in women."),
        ("Sex",
         "Male = 1 · Female = 0. Men generally have a higher risk of heart disease compared to pre-menopausal women."),
        ("Chest Pain Type",
         "0 = Typical Angina (exertion-induced chest pain) · 1 = Atypical Angina · 2 = Non-Anginal Pain (not heart-related) · 3 = Asymptomatic (no pain, but disease may be present)"),
        ("Resting BP",
         "Blood pressure measured at rest (mm Hg). Normal is ~120/80. Values above 140 indicate hypertension and elevated cardiac risk."),
        ("Cholesterol",
         "Serum cholesterol level (mg/dl). Below 200 is normal, 240+ is high. Elevated cholesterol contributes to arterial plaque buildup."),
        ("Fasting Blood Sugar",
         "Is fasting blood sugar > 120 mg/dl? Yes = 1. A marker of diabetes, which significantly increases heart disease risk."),
        ("Resting ECG",
         "0 = Normal · 1 = ST-T wave abnormality (irregular cardiac electrical activity) · 2 = Left Ventricular Hypertrophy (thickened heart wall)"),
        ("Max Heart Rate",
         "Peak heart rate achieved during exercise. A lower maximum may indicate reduced cardiac output and higher disease likelihood."),
        ("Exercise-Induced Angina",
         "Chest pain occurring during physical exertion. Yes = strong risk indicator — suggests inadequate blood supply to the heart during activity."),
        ("Oldpeak (ST Depression)",
         "ST segment depression on ECG post-exercise vs. rest. 0 = normal. Higher values reflect greater cardiac stress and ischemia."),
        ("ST Slope",
         "0 = Upsloping (healthy) · 1 = Flat (moderate risk) · 2 = Downsloping (high risk — indicates reduced myocardial blood flow)"),
        ("Major Vessels",
         "Number of major coronary vessels (0–3) showing blockage on fluoroscopy. More blocked vessels = higher disease severity."),
        ("Thal",
         "Thalassemia test result. 0/1 = Normal · 2 = Fixed Defect (permanent myocardial damage) · 3 = Reversible Defect (temporary blood flow reduction)"),
    ]

    for name, desc in fields:
        st.markdown(f"""
        <div class="guide-card">
            <div class="fname">{name}</div>
            <div class="fdesc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)