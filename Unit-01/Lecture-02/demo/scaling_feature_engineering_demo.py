from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def minmax_scale(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if mx == mn:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def zscore_scale(s: pd.Series) -> pd.Series:
    mu = float(s.mean())
    sd = float(s.std(ddof=0))  # population std for standardization
    if sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


def main() -> None:
    base = Path(".")
    data_path = base / "data" / "student_features_raw.csv"
    out_path = base / "data" / "student_features_engineered.csv"
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce")

    print("=== Raw Dataset ===")
    print(df.head(5).to_string(index=False))

    # --- Scaling demo ---
    scale_cols = ["attendance_pct", "study_hours_week", "family_income_k"]
    mm = df[scale_cols].apply(minmax_scale)
    zz = df[scale_cols].apply(zscore_scale)

    # Show how distance can be dominated by a large-scale feature (income).
    a = df.loc[df["student_id"] == 2005, scale_cols].to_numpy(dtype=float).ravel()
    b = df.loc[df["student_id"] == 2011, scale_cols].to_numpy(dtype=float).ravel()
    d_raw = euclidean(a, b)

    a_mm = mm.loc[df["student_id"] == 2005, :].to_numpy(dtype=float).ravel()
    b_mm = mm.loc[df["student_id"] == 2011, :].to_numpy(dtype=float).ravel()
    d_mm = euclidean(a_mm, b_mm)

    a_zz = zz.loc[df["student_id"] == 2005, :].to_numpy(dtype=float).ravel()
    b_zz = zz.loc[df["student_id"] == 2011, :].to_numpy(dtype=float).ravel()
    d_zz = euclidean(a_zz, b_zz)

    print("\n=== Scaling Effect on Distance (Students 2005 vs 2011) ===")
    print("Features used:", scale_cols)
    print("Raw distance:     ", round(d_raw, 3))
    print("Min-max distance: ", round(d_mm, 3))
    print("Z-score distance: ", round(d_zz, 3))

    # --- Feature engineering demo ---
    feat = df.copy()
    feat["join_month"] = feat["join_date"].dt.month
    feat["join_weekday"] = feat["join_date"].dt.day_name()

    feat["attendance_bucket"] = pd.cut(
        feat["attendance_pct"],
        bins=[0, 80, 90, 100],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    feat["income_log1p"] = np.log1p(feat["family_income_k"])
    feat["effort_index"] = feat["study_hours_week"] * (feat["attendance_pct"] / 100.0)
    feat["has_backlog"] = (feat["backlogs"] > 0).astype(int)

    # One-hot encoding (basic) for program.
    feat = pd.get_dummies(feat, columns=["program"], prefix="program", dtype=int)

    feat.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    print("\nSaved engineered dataset:", out_path)

    # --- Visuals ---
    # Income vs log-income histogram.
    plt.figure(figsize=(9, 3.5))
    plt.subplot(1, 2, 1)
    plt.hist(df["family_income_k"], bins=8, edgecolor="black")
    plt.title("Income (INR thousands)")
    plt.xlabel("income_k")
    plt.ylabel("count")

    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df["family_income_k"]), bins=8, edgecolor="black")
    plt.title("log1p(Income)")
    plt.xlabel("log1p(income_k)")
    plt.tight_layout()
    plt.savefig(images_dir / "income_vs_log_income.png", dpi=200)
    plt.close()

    # Scatter: study hours vs CGPA (just to see relationship).
    plt.figure(figsize=(5.5, 3.5))
    plt.scatter(df["study_hours_week"], df["cgpa"])
    plt.title("Study Hours vs CGPA")
    plt.xlabel("study_hours_week")
    plt.ylabel("cgpa")
    plt.tight_layout()
    plt.savefig(images_dir / "hours_vs_cgpa.png", dpi=200)
    plt.close()

    print("Saved plots into:", images_dir)


if __name__ == "__main__":
    main()

