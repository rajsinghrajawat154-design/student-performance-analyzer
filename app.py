"""Streamlit web app for the Student Performance Analyzer & Score Predictor.
Run locally:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from main import (FEATURES, PASS_MARK, TARGET, add_features, clean_data,
                  load_data, train_model)

DATA_FILE = Path(__file__).parent / "data" / "students.csv"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


@st.cache_data
def get_data():
    df, info = clean_data(load_data(DATA_FILE))
    return add_features(df), info


@st.cache_resource
def get_model(df):
    return train_model(df)


df, info = get_data()
model, metrics, n_train, n_test = get_model(df)

st.title("🎓 Student Performance Analyzer & Score Predictor")
st.write(
    "Built with **Python, Pandas and NumPy**. The prediction model is a linear regression "
    "written from scratch with NumPy (no scikit-learn). "
    "The dataset is synthetic, generated for this project."
)

# ---------- Sidebar: inputs ----------
st.sidebar.header("Enter student details")
hours = st.sidebar.slider("Study hours per day", 0.0, 10.0, 5.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
prev = st.sidebar.slider("Previous exam score", 0, 100, 60)
sleep = st.sidebar.slider("Sleep hours per day", 3.0, 10.0, 7.0, 0.5)

features = np.array([[hours, attendance, prev, sleep]], dtype=float)
score = float(np.clip(model.predict(features)[0], 0, 100))
passed = score >= PASS_MARK

# ---------- Prediction ----------
st.subheader("Predicted final score")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted score", f"{score:.1f} / 100")
c2.metric("Result", "Pass ✅" if passed else "Fail ❌")
c3.metric("Class average", f"{df[TARGET].mean():.1f}", delta=f"{score - df[TARGET].mean():+.1f} vs average")

st.divider()

# ---------- Insights ----------
left, right = st.columns(2)

with left:
    st.subheader("What affects final score?")
    st.caption("Correlation of each factor with the final score (higher = stronger link)")
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    st.bar_chart(corr)

    st.subheader("Marks gained per unit (model coefficients)")
    coef = pd.DataFrame({"Factor": FEATURES, "Marks per unit": np.round(model.coef_, 3)})
    st.dataframe(coef, hide_index=True, use_container_width=True)

with right:
    st.subheader("Average score by attendance")
    band = df.groupby("attendance_band", observed=True)[TARGET].mean().round(2)
    st.bar_chart(band)

    st.subheader("Model performance (unseen test data)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R2']:.2f}")
    m2.metric("MAE", f"{metrics['MAE']:.2f}")
    m3.metric("RMSE", f"{metrics['RMSE']:.2f}")
    st.caption(f"Trained on {n_train} students, tested on {n_test}.")

st.divider()

# ---------- Data ----------
with st.expander("Data cleaning summary"):
    st.write(
        f"Rows before: **{info['rows_before']}** → after: **{info['rows_after']}**  \n"
        f"Duplicates removed: **{info['duplicates_removed']}**  \n"
        f"Missing/invalid values filled with median: **{info['missing_values_filled']}**"
    )

with st.expander("View cleaned dataset"):
    st.dataframe(df, use_container_width=True)

st.caption("Source code: github.com/rajsinghrajawat154-design/student-performance-analyzer")
