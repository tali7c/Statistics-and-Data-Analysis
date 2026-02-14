from __future__ import annotations

import csv
import math
import statistics as stats
from pathlib import Path


def read_numeric_table(path: Path) -> dict[str, list[float]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {name: [] for name in (reader.fieldnames or [])}
        for row in reader:
            for k in cols:
                cols[k].append(float(row[k]))
    return cols


def quartiles_tukey(values: list[float]) -> tuple[float, float]:
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
    return float(q1), float(q3)


def moment_skewness_kurtosis(values: list[float]) -> tuple[float, float]:
    """Return (skewness g1, excess kurtosis). Population-moment style."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(values) / n
    devs = [v - mean for v in values]
    m2 = sum(d * d for d in devs) / n
    if m2 == 0:
        return 0.0, float("nan")
    m3 = sum(d**3 for d in devs) / n
    m4 = sum(d**4 for d in devs) / n
    g1 = m3 / (m2 ** 1.5)
    g2 = m4 / (m2**2)
    return float(g1), float(g2 - 3.0)


def summary(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    n = len(v)
    q1, q3 = quartiles_tukey(v) if n >= 2 else (float("nan"), float("nan"))
    skew, ex_kurt = moment_skewness_kurtosis(v) if n >= 2 else (float("nan"), float("nan"))
    return {
        "count": float(n),
        "mean": float(stats.mean(v)) if n else float("nan"),
        "median": float(stats.median(v)) if n else float("nan"),
        "std": float(stats.stdev(v)) if n >= 2 else 0.0,
        "min": float(min(v)) if n else float("nan"),
        "q1": float(q1),
        "q3": float(q3),
        "max": float(max(v)) if n else float("nan"),
        "skew": float(skew),
        "excess_kurtosis": float(ex_kurt),
    }


def try_plot_hist_grid(cols: dict[str, list[float]], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except Exception:
        print("matplotlib not installed; skipping plots.")
        return

    names = list(cols.keys())
    n = len(names)
    rows, cols_n = 2, 3
    fig, axes = plt.subplots(rows, cols_n, figsize=(10, 6))
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        vals = cols[name]
        bins = min(10, max(5, int(math.sqrt(len(vals)))))
        ax.hist(vals, bins=bins, edgecolor="black")
        ax.set_title(name)
        ax.grid(alpha=0.15)

    # Hide unused axes.
    for j in range(n, rows * cols_n):
        axes[j].axis("off")

    fig.suptitle("Histograms (Distribution Shapes) - Multi-feature Dataset", y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    base = Path(".")
    data_path = base / "data" / "multi_feature_distributions.csv"
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cols = read_numeric_table(data_path)

    print("=== Dimensional Summary (per feature) ===")
    header = (
        f"{'feature':<14} {'n':>3} {'mean':>7} {'median':>7} {'std':>7} {'min':>7} {'q1':>7} {'q3':>7} {'max':>7} {'skew':>7}"
    )
    print(header)
    print("-" * len(header))

    for name in cols:
        s = summary(cols[name])
        print(
            f"{name:<14} {int(s['count']):>3} {s['mean']:>7.2f} {s['median']:>7.2f} {s['std']:>7.2f} {s['min']:>7.2f} "
            f"{s['q1']:>7.2f} {s['q3']:>7.2f} {s['max']:>7.2f} {s['skew']:>7.2f}"
        )

    out_png = images_dir / "hists_grid.png"
    try_plot_hist_grid(cols, out_png)
    if out_png.exists():
        print("Saved plot:", out_png)

if __name__ == "__main__":
    main()
