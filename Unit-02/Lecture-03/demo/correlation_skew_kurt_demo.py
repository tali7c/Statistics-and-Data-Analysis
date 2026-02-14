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


def read_numeric_table(path: str):
    """Return dict[column] -> list[float] for a CSV with numeric columns."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for k in cols:
                cols[k].append(float(row[k]))
    return cols


def pearson_r(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 paired values for correlation")

    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxy = 0.0
    sx2 = 0.0
    sy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - xbar
        dy = y - ybar
        sxy += dx * dy
        sx2 += dx * dx
        sy2 += dy * dy

    denom = math.sqrt(sx2 * sy2)
    return sxy / denom if denom != 0 else float("nan")


def moment_skewness_kurtosis(values):
    """Moment (population-moment) skewness and kurtosis. Excess = kurtosis - 3."""
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 values")
    mean = sum(values) / n
    devs = [v - mean for v in values]
    m2 = sum(d * d for d in devs) / n
    m3 = sum(d**3 for d in devs) / n
    m4 = sum(d**4 for d in devs) / n
    if m2 == 0:
        return mean, 0.0, float("nan"), float("nan"), float("nan")
    g1 = m3 / (m2 ** 1.5)
    g2 = m4 / (m2**2)
    return mean, m2, g1, g2, g2 - 3.0


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    return plt


def maybe_plot_scatter(xs, ys, out_png: str, title: str, xlabel: str, ylabel: str):
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    plt.figure(figsize=(5.5, 3.5))
    plt.scatter(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()


def maybe_plot_hist(values, out_png: str, title: str, xlabel: str):
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    plt.figure(figsize=(6, 3.5))
    plt.hist(values, bins=min(10, max(4, int(len(values) / 2))), edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()


def maybe_plot_corr_heatmap(cols, out_png: str):
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    names = list(cols.keys())
    m = []
    for a in names:
        row = []
        for b in names:
            row.append(pearson_r(cols[a], cols[b]))
        m.append(row)

    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(m, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.title("Correlation Heatmap (Pearson r)")

    # Annotate cells (small matrix).
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{m[i][j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()


def main():
    base = Path(".")

    print("=== Correlation (Pearson r) ===")
    xs, ys = read_two_column_csv(str(base / "data" / "pairs_hours_score.csv"), "hours", "score")
    r1 = pearson_r(xs, ys)
    print(f"Hours vs Score: r = {r1:.4f}")
    maybe_plot_scatter(xs, ys, str(base / "images" / "hours_score_scatter.png"),
                       "Hours vs Score", "Hours", "Score")

    px, dy = read_two_column_csv(str(base / "data" / "pairs_price_demand.csv"), "price", "demand")
    r2 = pearson_r(px, dy)
    print(f"Price vs Demand: r = {r2:.4f}")
    maybe_plot_scatter(px, dy, str(base / "images" / "price_demand_scatter.png"),
                       "Price vs Demand", "Price", "Demand")

    nx, ny = read_two_column_csv(str(base / "data" / "nonlinear_x_x2.csv"), "x", "y")
    r3 = pearson_r(nx, ny)
    print(f"x vs x^2 (non-linear): r = {r3:.4f}")
    maybe_plot_scatter(nx, ny, str(base / "images" / "x_x2_scatter.png"),
                       "x vs x^2", "x", "y")

    print("\n=== Correlation Matrix (Student Metrics) ===")
    cols = read_numeric_table(str(base / "data" / "student_metrics.csv"))
    names = list(cols.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            print(f"{a} vs {b}: r = {pearson_r(cols[a], cols[b]):.4f}")
    maybe_plot_corr_heatmap(cols, str(base / "images" / "corr_heatmap.png"))

    print("\n=== Skewness and Kurtosis (Moment) ===")
    income = read_column_csv(str(base / "data" / "income_right_skew.csv"), "income")
    mean, m2, g1, g2, ex = moment_skewness_kurtosis(income)
    print("Income (right-skew example)")
    print(f"  n={len(income)} mean={mean:.2f} median={stats.median(income):.2f}")
    print(f"  skewness g1={g1:.3f}  excess kurtosis={ex:.3f}")
    maybe_plot_hist(income, str(base / "images" / "income_hist.png"),
                    "Income (Right-Skew)", "Income")

    left = read_column_csv(str(base / "data" / "scores_left_skew.csv"), "score")
    mean, m2, g1, g2, ex = moment_skewness_kurtosis(left)
    print("Scores (left-skew example)")
    print(f"  n={len(left)} mean={mean:.2f} median={stats.median(left):.2f}")
    print(f"  skewness g1={g1:.3f}  excess kurtosis={ex:.3f}")
    maybe_plot_hist(left, str(base / "images" / "left_skew_hist.png"),
                    "Scores (Left-Skew)", "Score")

    sym = read_column_csv(str(base / "data" / "symmetric_small.csv"), "value")
    mean, m2, g1, g2, ex = moment_skewness_kurtosis(sym)
    print("Symmetric small example")
    print(f"  n={len(sym)} mean={mean:.2f} median={stats.median(sym):.2f}")
    print(f"  skewness g1={g1:.3f}  excess kurtosis={ex:.3f}")
    maybe_plot_hist(sym, str(base / "images" / "symmetric_hist.png"),
                    "Symmetric Example", "Value")

    print("\nDone. If matplotlib is installed, plots are saved in the images/ folder.")


if __name__ == "__main__":
    main()
