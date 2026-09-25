
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data"/"student_data.csv"
MODELS = ROOT/"models"

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    return (
        joblib.load(MODELS/"placement_model.pkl"),
        joblib.load(MODELS/"package_model.pkl"),
        joblib.load(MODELS/"features.pkl"),
        joblib.load(MODELS/"metrics.pkl")
    )

clf, reg, features, metrics = load_models()
df = pd.read_csv(DATA)

st.title("🎓 Smart Student Performance & Placement Predictor")
st.caption("Intermediate Data Science Project • Educational demonstration using synthetic data")

with st.sidebar:
    st.header("Student Profile")
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
    attendance = st.slider("Attendance (%)", 50, 100, 80)
    internships = st.slider("Internships", 0, 3, 1)
    projects = st.slider("Projects", 0, 5, 2)
    dsa = st.slider("DSA Score", 0, 100, 70)
    communication = st.slider("Communication Score", 0, 100, 75)
    certifications = st.slider("Certifications", 0, 8, 2)
    aptitude = st.slider("Aptitude Score", 0, 100, 70)

X = pd.DataFrame([[cgpa,attendance,internships,projects,dsa,communication,certifications,aptitude]], columns=features)
prob = float(clf.predict_proba(X)[0,1])
pred = int(prob >= 0.5)
pkg = float(reg.predict(X)[0])

c1,c2,c3 = st.columns(3)
c1.metric("Placement Probability", f"{prob*100:.1f}%")
c2.metric("Prediction", "Likely Placed" if pred else "Needs Improvement")
c3.metric("Estimated Package", f"₹{pkg:.2f} LPA" if pred else "N/A")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Insights","🤖 Model Performance","📁 Dataset"])

with tab1:
    st.subheader("Your Profile")
    st.dataframe(X.T.rename(columns={0:"Value"}), use_container_width=True)
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8,4))
    imp.plot(kind="barh", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    st.subheader("Personalized Improvement Areas")
    advice=[]
    if cgpa < 7.0: advice.append("Improve CGPA through consistent semester preparation.")
    if dsa < 65: advice.append("Practice DSA regularly and solve timed problems.")
    if projects < 2: advice.append("Build at least 2–3 demonstrable projects.")
    if internships == 0: advice.append("Look for internship or real-world project experience.")
    if communication < 65: advice.append("Practice technical communication and mock interviews.")
    if aptitude < 65: advice.append("Work on quantitative, logical and verbal aptitude.")
    if not advice: advice.append("Maintain your current profile and keep building interview readiness.")
    for a in advice: st.write("•", a)

with tab2:
    st.subheader("Classification Metrics")
    m = metrics["classification"]
    cols = st.columns(5)
    for col,(k,v) in zip(cols,m.items()):
        col.metric(k.upper().replace("_"," "), f"{v:.3f}")
    st.info("Metrics are calculated on a held-out test split of the synthetic dataset.")
    st.subheader("Package Regression")
    rm = metrics["regression"]
    st.write(f"MAE: **{rm['mae']:.2f} LPA**")
    st.write(f"R²: **{rm['r2']:.3f}**")

with tab3:
    st.write(f"Rows: {len(df):,} • Columns: {len(df.columns)}")
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "student_data.csv", "text/csv")

st.divider()
st.caption("⚠️ This is an educational ML demo. Predictions should not be used as the sole basis for real hiring or career decisions.")
