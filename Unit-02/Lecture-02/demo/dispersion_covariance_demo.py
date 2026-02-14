import csv
import math
import statistics as stats
from pathlib import Path


def read_column_csv(path: str, column: str):
    values = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def read_two_column_csv(path: str, col_x: str, col_y: str):
    xs, ys = [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row[col_x]))
            ys.append(float(row[col_y]))
    return xs, ys


def quartiles(values):
    """Q1/Q3 using the median-of-halves (Tukey-style) method."""
    v = sorted(values)
    n = len(v)
    if n < 2:
        raise ValueError("Need at least 2 values for quartiles")

    mid = n // 2
    if n % 2 == 0:
        lower = v[:mid]
        upper = v[mid:]
    else:
        lower = v[:mid]
        upper = v[mid + 1 :]

    q1 = stats.median(lower)
    q3 = stats.median(upper)
    return q1, q3


def iqr_fences(q1, q3):
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def covariance_sample(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 paired values for covariance")
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    s = 0.0
    for x, y in zip(xs, ys):
        s += (x - xbar) * (y - ybar)
    return s / (n - 1)


def maybe_plot_scores(values, out_png: str):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not installed; skipping plots.")
        return

    plt.figure(figsize=(6, 3.5))
    plt.hist(values, bins=6, edgecolor="black")
    plt.title("Scores: Histogram")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()


def maybe_plot_scatter(xs, ys, out_png: str, title: str, xlabel: str, ylabel: str):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    plt.figure(figsize=(5.5, 3.5))
    plt.scatter(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()


def main():
    base = Path(".")

    scores_path = base / "data" / "scores_small.csv"
    scores = read_column_csv(str(scores_path), "score")
    n = len(scores)

    mean = stats.mean(scores)
    median = stats.median(scores)
    mode = stats.mode(scores)
    r = max(scores) - min(scores)
    q1, q3 = quartiles(scores)
    iqr = q3 - q1
    s2 = stats.variance(scores)  # sample variance (n-1)
    s = stats.stdev(scores)      # sample std dev
    cv = (s / mean) * 100 if mean != 0 else float("nan")

    print("=== Dispersion Demo ===")
    print(f"Scores (n={n}): {sorted(scores)}")
    print(f"Mean:   {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode:   {mode:.2f}")
    print(f"Range:  {r:.2f}")
    print(f"Q1:     {q1:.2f}")
    print(f"Q3:     {q3:.2f}")
    print(f"IQR:    {iqr:.2f}")
    print(f"s^2:    {s2:.2f}")
    print(f"s:      {s:.2f}")
    print(f"CV (%):  {cv:.2f}")

    # Simple z-score examples (using sample stdev).
    if s != 0:
        z_min = (min(scores) - mean) / s
        z_max = (max(scores) - mean) / s
        print(f"z(min={min(scores):.0f}): {z_min:.2f}")
        print(f"z(max={max(scores):.0f}): {z_max:.2f}")

    maybe_plot_scores(scores, str(base / "images" / "scores_hist.png"))

    # Optional: outlier detection example (IQR rule)
    incomes_path = base / "data" / "incomes_outlier.csv"
    if incomes_path.exists():
        incomes = read_column_csv(str(incomes_path), "income")
        iq1, iq3 = quartiles(incomes)
        low_f, up_f = iqr_fences(iq1, iq3)
        outliers = [v for v in incomes if v < low_f or v > up_f]
        print("\n=== IQR Outlier Check (Income) ===")
        print(f"Incomes: {sorted(incomes)}")
        print(f"Q1: {iq1:.2f}  Q3: {iq3:.2f}  IQR: {(iq3 - iq1):.2f}")
        print(f"Fences: [{low_f:.2f}, {up_f:.2f}]")
        print(f"Outliers: {outliers if outliers else 'None'}")

    print("\n=== Covariance Demo ===")
    xs, ys = read_two_column_csv(str(base / "data" / "pairs_hours_score.csv"), "hours", "score")
    cov_xy = covariance_sample(xs, ys)
    print(f"Hours vs Score covariance (sample): {cov_xy:.2f}")
    maybe_plot_scatter(xs, ys, str(base / "images" / "hours_score_scatter.png"),
                       "Hours vs Score", "Hours", "Score")

    px, dy = read_two_column_csv(str(base / "data" / "pairs_price_demand.csv"), "price", "demand")
    cov_pd = covariance_sample(px, dy)
    print(f"Price vs Demand covariance (sample): {cov_pd:.2f}")
    maybe_plot_scatter(px, dy, str(base / "images" / "price_demand_scatter.png"),
                       "Price vs Demand", "Price", "Demand")

    print("\nDone. If matplotlib is installed, plots are saved in the images/ folder.")


if __name__ == "__main__":
    main()
