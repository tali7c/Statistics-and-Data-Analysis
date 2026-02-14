from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a GUI window.
import matplotlib.pyplot as plt  # noqa: E402


def missing_report(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).sort_values(ascending=False)


def main() -> None:
    base = Path(".")
    raw_path = base / "data" / "messy_students.csv"
    out_clean = base / "data" / "students_clean.csv"
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    print("=== Raw Data ===")
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head(6).to_string(index=False))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissingness (%):")
    print(missing_report(df).round(1).to_string())

    dup_count = int(df.duplicated(subset=["student_id"]).sum())
    print(f"\nDuplicate student_id rows: {dup_count}")
    print("\nUnique program values (raw):", sorted(df["program"].dropna().astype(str).unique()))
    print("Unique gender values (raw):", sorted(df["gender"].dropna().astype(str).unique()))

    # --- Cleaning starts here ---
    clean = df.copy()

    # Normalize string columns.
    for col in ["gender", "program", "city"]:
        clean[col] = clean[col].astype("string").str.strip()

    clean["program"] = clean["program"].str.upper()

    gender_map = {
        "M": "M",
        "MALE": "M",
        "F": "F",
        "FEMALE": "F",
    }
    clean["gender"] = clean["gender"].str.upper().map(gender_map)

    # Coerce numeric columns.
    numeric_cols = ["age", "cgpa", "attendance_pct", "study_hours_week"]
    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # Parse dates with multiple common formats (mixed-format columns are common in exports).
    # Important: do NOT globally set dayfirst=True, because it can misinterpret ISO dates (YYYY-MM-DD).
    join_raw = clean["join_date"].astype("string").str.strip()
    dt_iso = pd.to_datetime(join_raw, format="%Y-%m-%d", errors="coerce")
    dt_dmy = pd.to_datetime(join_raw, format="%d-%m-%Y", errors="coerce")
    dt_slash = pd.to_datetime(join_raw, format="%Y/%m/%d", errors="coerce")
    clean["join_date"] = dt_iso.fillna(dt_dmy).fillna(dt_slash)

    # Flag out-of-range values as missing (then impute with median).
    clean.loc[~clean["age"].between(16, 30), "age"] = np.nan
    clean.loc[~clean["cgpa"].between(0, 10), "cgpa"] = np.nan
    clean.loc[~clean["attendance_pct"].between(0, 100), "attendance_pct"] = np.nan
    clean.loc[~clean["study_hours_week"].between(0, 60), "study_hours_week"] = np.nan

    # Simple imputation for numeric columns (median is robust to outliers).
    for col in numeric_cols:
        med = float(clean[col].median())
        clean[col] = clean[col].fillna(med)

    # Remove duplicates by student_id (keep the first occurrence).
    before = len(clean)
    clean = clean.drop_duplicates(subset=["student_id"], keep="first")
    after = len(clean)

    print("\n=== Cleaned Data ===")
    print("Rows after dedup:", after, f"(removed {before - after})")
    print("\nUnique program values (clean):", sorted(clean["program"].dropna().astype(str).unique()))
    print("Unique gender values (clean):", sorted(clean["gender"].dropna().astype(str).unique()))
    print("\nMissingness (%):")
    print(missing_report(clean).round(1).to_string())

    invalid_dates = int(clean["join_date"].isna().sum())
    print(f"\nInvalid/unparsed join_date values remaining: {invalid_dates}")

    # Save cleaned dataset.
    clean.to_csv(out_clean, index=False, date_format="%Y-%m-%d")
    print("\nSaved:", out_clean)

    # --- Visuals ---
    # Missingness before/after.
    for name, rep in [("before", missing_report(df)), ("after", missing_report(clean))]:
        plt.figure(figsize=(7, 3.5))
        rep.plot(kind="bar")
        plt.ylabel("Missing (%)")
        plt.title(f"Missingness ({name})")
        plt.tight_layout()
        plt.savefig(images_dir / f"missingness_{name}.png", dpi=200)
        plt.close()

    # Attendance boxplot before/after (shows outliers clearly).
    for name, series in [("before", df["attendance_pct"]), ("after", clean["attendance_pct"])]:
        plt.figure(figsize=(5, 3.5))
        vals = pd.to_numeric(series, errors="coerce").dropna()
        plt.boxplot(vals, vert=True)
        plt.ylabel("Attendance (%)")
        plt.title(f"Attendance Boxplot ({name})")
        plt.tight_layout()
        plt.savefig(images_dir / f"attendance_box_{name}.png", dpi=200)
        plt.close()

    print("Saved plots into:", images_dir)


if __name__ == "__main__":
    main()
