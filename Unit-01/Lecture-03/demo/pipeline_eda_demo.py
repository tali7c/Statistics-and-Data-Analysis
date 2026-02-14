from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def missing_report(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).sort_values(ascending=False)


def main() -> None:
    base = Path(".")
    raw_path = base / "data" / "case_study.csv"
    clean_path = base / "data" / "case_study_clean.csv"
    by_prog_path = base / "data" / "summary_by_program.csv"
    corr_path = base / "data" / "corr_matrix.csv"

    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    print("=== Raw Data ===")
    print("Shape:", df.shape)
    print(df.head(6).to_string(index=False))
    print("\nMissingness (%):")
    print(missing_report(df).round(1).to_string())

    # --- Pipeline: clean + validate ---
    clean = df.copy()

    # Standardize strings.
    for col in ["program", "gender", "hostel"]:
        clean[col] = clean[col].astype("string").str.strip()
    clean["program"] = clean["program"].str.upper()
    clean["gender"] = clean["gender"].str.upper()
    clean["hostel"] = clean["hostel"].str.upper().map({"YES": 1, "NO": 0})

    # Convert numeric and parse date.
    num_cols = ["attendance_pct", "study_hours_week", "cgpa", "final_marks"]
    for col in num_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean["join_date"] = pd.to_datetime(clean["join_date"], errors="coerce")

    # Range checks (flag as missing, then impute).
    clean.loc[~clean["attendance_pct"].between(0, 100), "attendance_pct"] = np.nan
    clean.loc[~clean["study_hours_week"].between(0, 60), "study_hours_week"] = np.nan
    clean.loc[~clean["cgpa"].between(0, 10), "cgpa"] = np.nan
    clean.loc[~clean["final_marks"].between(0, 100), "final_marks"] = np.nan

    # Median imputation for numeric columns.
    for col in num_cols:
        clean[col] = clean[col].fillna(float(clean[col].median()))

    clean.to_csv(clean_path, index=False, date_format="%Y-%m-%d")
    print("\nSaved cleaned dataset:", clean_path)

    # --- EDA ---
    print("\n=== EDA: Cleaned Data ===")
    print("Missingness (%):")
    print(missing_report(clean).round(1).to_string())

    print("\nDescribe (numeric):")
    print(clean[num_cols].describe().round(2).to_string())

    by_prog = (
        clean.groupby("program")[num_cols]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
    )
    by_prog.to_csv(by_prog_path)
    print("\nSaved group summary:", by_prog_path)

    corr = clean[num_cols].corr().round(3)
    corr.to_csv(corr_path)
    print("Saved correlation matrix:", corr_path)

    # --- Plots ---
    # Final marks histogram.
    plt.figure(figsize=(6, 3.5))
    plt.hist(clean["final_marks"], bins=8, edgecolor="black")
    plt.title("Final Marks: Histogram (Cleaned)")
    plt.xlabel("final_marks")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(images_dir / "final_marks_hist.png", dpi=200)
    plt.close()

    # Boxplot final marks by program.
    plt.figure(figsize=(6, 3.5))
    programs = sorted(clean["program"].dropna().unique())
    data = [clean.loc[clean["program"] == p, "final_marks"].to_numpy() for p in programs]
    plt.boxplot(data, labels=programs)
    plt.title("Final Marks by Program")
    plt.xlabel("program")
    plt.ylabel("final_marks")
    plt.tight_layout()
    plt.savefig(images_dir / "final_marks_by_program.png", dpi=200)
    plt.close()

    # Scatter: study hours vs final marks.
    plt.figure(figsize=(5.5, 3.5))
    plt.scatter(clean["study_hours_week"], clean["final_marks"])
    plt.title("Study Hours vs Final Marks")
    plt.xlabel("study_hours_week")
    plt.ylabel("final_marks")
    plt.tight_layout()
    plt.savefig(images_dir / "hours_vs_marks.png", dpi=200)
    plt.close()

    # Correlation heatmap.
    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation Heatmap (Numeric)")
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(images_dir / "corr_heatmap.png", dpi=200)
    plt.close()

    print("Saved plots into:", images_dir)


if __name__ == "__main__":
    main()

