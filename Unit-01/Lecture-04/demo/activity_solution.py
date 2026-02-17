from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    base = Path(".")
    raw_path = base / "data" / "campus_cafe_transactions.csv"
    clean_path = base / "data" / "campus_cafe_clean.csv"

    rev_cat_path = base / "data" / "revenue_by_category.csv"
    rev_pay_path = base / "data" / "revenue_by_payment.csv"
    top_path = base / "data" / "top_transactions.csv"
    daily_path = base / "data" / "daily_revenue.csv"

    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    # Basic type conversions.
    for col in ["category", "payment_mode"]:
        df[col] = df[col].astype("string").str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units"] = pd.to_numeric(df["units"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0.0)

    # Cleaning rules:
    # - negative units are invalid => drop the row.
    # - missing units => impute with median units.
    # - discount outside [0, 50] => cap to [0, 50].
    # - unit_price outliers (>200) => replace with category median (computed from valid prices).
    # Drop rows with negative units (keep NaN for later imputation).
    df = df[(df["units"].isna()) | (df["units"] >= 0)]
    median_units = float(df["units"].median())
    df["units"] = df["units"].fillna(median_units)

    df["discount_pct"] = df["discount_pct"].clip(lower=0, upper=50)

    valid_price = df["unit_price"].between(1, 200)
    cat_median_price = (
        df.loc[valid_price]
        .groupby("category")["unit_price"]
        .median()
        .to_dict()
    )
    def fix_price(row):
        p = row["unit_price"]
        if pd.isna(p) or p <= 0 or p > 200:
            return float(cat_median_price.get(row["category"], np.nan))
        return float(p)

    df["unit_price"] = df.apply(fix_price, axis=1)

    # Drop rows that still cannot be used.
    df = df.dropna(subset=["date", "units", "unit_price"])

    # Feature engineering.
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["gross_amount"] = df["units"] * df["unit_price"]
    df["net_amount"] = df["gross_amount"] * (1.0 - df["discount_pct"] / 100.0)

    # Save cleaned dataset.
    df.to_csv(clean_path, index=False, date_format="%Y-%m-%d")

    # Summary tables.
    rev_by_cat = df.groupby("category")["net_amount"].agg(["count", "sum", "mean"]).round(2)
    rev_by_pay = df.groupby("payment_mode")["net_amount"].agg(["count", "sum", "mean"]).round(2)
    daily = df.groupby(df["date"].dt.date)["net_amount"].sum().round(2).rename("net_revenue").reset_index()

    top = df.sort_values("net_amount", ascending=False).head(5)[
        ["txn_id", "date", "category", "payment_mode", "units", "unit_price", "discount_pct", "net_amount"]
    ].round(2)

    rev_by_cat.to_csv(rev_cat_path)
    rev_by_pay.to_csv(rev_pay_path)
    daily.to_csv(daily_path, index=False)
    top.to_csv(top_path, index=False)

    print("Saved:", clean_path)
    print("Saved:", rev_cat_path)
    print("Saved:", rev_pay_path)
    print("Saved:", daily_path)
    print("Saved:", top_path)

    # Plots.
    plt.figure(figsize=(6, 3.5))
    rev_by_cat["sum"].plot(kind="bar")
    plt.title("Net Revenue by Category")
    plt.ylabel("net revenue (INR)")
    plt.tight_layout()
    plt.savefig(images_dir / "revenue_by_category.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.hist(df["net_amount"], bins=10, edgecolor="black")
    plt.title("Net Amount per Transaction (Histogram)")
    plt.xlabel("net_amount (INR)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(images_dir / "net_amount_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 3.5))
    plt.plot(pd.to_datetime(daily["date"]), daily["net_revenue"], marker="o")
    plt.title("Daily Net Revenue")
    plt.xlabel("date")
    plt.ylabel("net revenue (INR)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(images_dir / "daily_revenue.png", dpi=200)
    plt.close()

    print("Saved plots into:", images_dir)


if __name__ == "__main__":
    main()
