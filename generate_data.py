"""Generates a realistic *synthetic* student dataset (with messy values)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

df = pd.DataFrame({
    "student_id": np.arange(1001, 1001 + n),
    "study_hours": rng.uniform(0.5, 8, n).round(1),
    "attendance": rng.uniform(50, 100, n).round(1),
    "prev_score": rng.normal(60, 12, n).clip(20, 100).round(1),
    "sleep_hours": rng.uniform(4, 9, n).round(1),
})

noise = rng.normal(0, 4, n)
df["final_score"] = (
    5 + 4.2 * df["study_hours"] + 0.15 * df["attendance"]
    + 0.35 * df["prev_score"] + 0.8 * df["sleep_hours"] + noise
).clip(0, 100).round(1)

# Make the data messy on purpose so the cleaning step has real work to do
df.loc[rng.choice(n, 10, replace=False), "sleep_hours"] = np.nan
df.loc[rng.choice(n, 8, replace=False), "attendance"] = np.nan
df.loc[rng.choice(n, 4, replace=False), "attendance"] = 130  # invalid value (>100%)
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)  # duplicate rows

df.to_csv("data/students.csv", index=False)
print(f"Saved data/students.csv with {len(df)} rows")
