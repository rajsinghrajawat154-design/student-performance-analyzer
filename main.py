"""Student Performance Analyzer & Score Predictor
Uses only pandas + NumPy: data cleaning, analysis, and a linear regression model built from scratch.

Usage:
    python main.py                              # full pipeline + report
    python main.py --predict 6 90 70 7          # predict score: study_hours attendance prev_score sleep_hours
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/students.csv")
OUT_DIR = Path("output")
FEATURES = ["study_hours", "attendance", "prev_score", "sleep_hours"]
TARGET = "final_score"
PASS_MARK = 40


# ---------- 1. Load & clean ----------
def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python generate_data.py")
    return pd.read_csv(path)


def clean_data(df):
    info = {"rows_before": len(df)}
    df = df.drop_duplicates(subset="student_id").copy()
    info["duplicates_removed"] = info["rows_before"] - len(df)

    df.loc[df["attendance"] > 100, "attendance"] = np.nan  # invalid percentages
    info["missing_values_filled"] = int(df[FEATURES + [TARGET]].isna().sum().sum())
    for col in FEATURES + [TARGET]:
        df[col] = df[col].fillna(df[col].median())

    info["rows_after"] = len(df)
    return df.reset_index(drop=True), info


# ---------- 2. Feature engineering & analysis ----------
def add_features(df):
    df = df.copy()
    df["attendance_band"] = pd.cut(
        df["attendance"], bins=[0, 60, 75, 90, 100],
        labels=["<60%", "60-75%", "75-90%", "90-100%"], include_lowest=True)
    df["grade"] = pd.cut(
        df[TARGET], bins=[-1, 39.99, 54.99, 69.99, 84.99, 100],
        labels=["F", "C", "B", "A", "A+"])
    df["result"] = np.where(df[TARGET] >= PASS_MARK, "Pass", "Fail")
    return df


def analyze(df):
    lines = ["=== DATASET SUMMARY ===", df[FEATURES + [TARGET]].describe().round(2).to_string(), ""]
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    lines += ["=== CORRELATION WITH FINAL SCORE ===", corr.round(3).to_string(), ""]
    band = df.groupby("attendance_band", observed=True)[TARGET].agg(["count", "mean"]).round(2)
    lines += ["=== AVERAGE SCORE BY ATTENDANCE BAND ===", band.to_string(), ""]
    grades = df["grade"].value_counts().sort_index().to_string()
    pass_rate = (df["result"] == "Pass").mean() * 100
    lines += ["=== GRADE DISTRIBUTION ===", grades, f"\nPass rate: {pass_rate:.1f}%", ""]
    return "\n".join(lines)


# ---------- 3. Linear regression from scratch (NumPy) ----------
class LinearRegressionNP:
    def fit(self, X, y):
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)
        A = np.c_[np.ones(len(X)), (X - self.mean_) / self.std_]
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        self.coef_ = w[1:] / self.std_
        self.intercept_ = w[0] - np.sum(self.coef_ * self.mean_)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def train_test_split(X, y, test_size=0.2, seed=42):
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def evaluate(y_true, y_pred):
    err = y_true - y_pred
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)), "R2": 1 - ss_res / ss_tot}


def train_model(df):
    X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    model = LinearRegressionNP().fit(X_tr, y_tr)
    return model, evaluate(y_te, model.predict(X_te)), len(X_tr), len(X_te)


# ---------- 4. Run ----------
def run_pipeline():
    df, info = clean_data(load_data())
    df = add_features(df)
    model, metrics, n_tr, n_te = train_model(df)

    report = ["=== DATA CLEANING ===", *[f"{k}: {v}" for k, v in info.items()], "", analyze(df),
              f"=== MODEL (train={n_tr}, test={n_te}) ===",
              *[f"{f}: {c:+.3f} marks per unit" for f, c in zip(FEATURES, model.coef_)],
              f"intercept: {model.intercept_:.3f}",
              *[f"{k}: {v:.3f}" for k, v in metrics.items()]]
    text = "\n".join(report)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "report.txt").write_text(text)
    df.to_csv(OUT_DIR / "cleaned_students.csv", index=False)
    print(text)
    print(f"\nSaved: {OUT_DIR/'report.txt'} and {OUT_DIR/'cleaned_students.csv'}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Student Performance Analyzer & Predictor")
    parser.add_argument("--predict", nargs=4, type=float, metavar=("HOURS", "ATTEND", "PREV", "SLEEP"),
                        help="predict final score from study_hours attendance prev_score sleep_hours")
    args = parser.parse_args()

    model = run_pipeline() if not args.predict else train_model(add_features(clean_data(load_data())[0]))[0]
    if args.predict:
        score = float(np.clip(model.predict(np.array([args.predict]))[0], 0, 100))
        print(f"Predicted final score: {score:.1f} ({'Pass' if score >= PASS_MARK else 'Fail'})")


if __name__ == "__main__":
    main()
