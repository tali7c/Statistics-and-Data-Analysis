from __future__ import annotations

import csv
import statistics as stats
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def quartiles_tukey(values: list[float]) -> tuple[float, float]:
    """Q1/Q3 using median-of-halves (Tukey-style) method."""
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


def summary(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    n = len(v)
    q1, q3 = quartiles_tukey(v) if n >= 2 else (float("nan"), float("nan"))
    return {
        "count": float(n),
        "mean": float(stats.mean(v)) if n else float("nan"),
        "std": float(stats.stdev(v)) if n >= 2 else 0.0,
        "min": float(min(v)) if n else float("nan"),
        "q1": float(q1),
        "median": float(stats.median(v)) if n else float("nan"),
        "q3": float(q3),
        "max": float(max(v)) if n else float("nan"),
    }


def try_plot_mean_by_program(means: dict[str, float], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except Exception:
        print("matplotlib not installed; skipping plot.")
        return

    programs = list(means.keys())
    ys = [means[p] for p in programs]

    plt.figure(figsize=(6.2, 3.6))
    bars = plt.bar(programs, ys)
    plt.title("Mean Final Score by Program")
    plt.ylabel("mean(final_score)")
    plt.ylim(0, max(ys) * 1.15)
    for b, y in zip(bars, ys):
        plt.text(b.get_x() + b.get_width() / 2, y + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    base = Path(".")
    data_path = base / "data" / "student_summary.csv"
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(data_path)

    numeric_cols = ["attendance_pct", "study_hours", "quiz_score", "final_score"]
    cols: dict[str, list[float]] = {c: [] for c in numeric_cols}
    programs: dict[str, list[float]] = {}

    for r in rows:
        prog = (r.get("program") or "").strip()
        programs.setdefault(prog, [])
        programs[prog].append(float(r["final_score"]))
        for c in numeric_cols:
            cols[c].append(float(r[c]))

    print("=== Overall Summary (like describe) ===")
    header = f"{'col':<16} {'n':>4} {'mean':>7} {'std':>7} {'min':>7} {'q1':>7} {'med':>7} {'q3':>7} {'max':>7}"
    print(header)
    print("-" * len(header))
    overall_rows: list[dict[str, str]] = []
    for c in numeric_cols:
        s = summary(cols[c])
        print(
            f"{c:<16} {int(s['count']):>4} {s['mean']:>7.2f} {s['std']:>7.2f} {s['min']:>7.2f} "
            f"{s['q1']:>7.2f} {s['median']:>7.2f} {s['q3']:>7.2f} {s['max']:>7.2f}"
        )
        overall_rows.append(
            {
                "column": c,
                "count": str(int(s["count"])),
                "mean": f"{s['mean']:.4f}",
                "std": f"{s['std']:.4f}",
                "min": f"{s['min']:.4f}",
                "q1": f"{s['q1']:.4f}",
                "median": f"{s['median']:.4f}",
                "q3": f"{s['q3']:.4f}",
                "max": f"{s['max']:.4f}",
            }
        )

    print("\n=== Grouped Summary: final_score by program ===")
    print(f"{'program':<8} {'n':>4} {'mean':>7} {'median':>7} {'std':>7}")
    print("-" * 38)
    by_prog_rows: list[dict[str, str]] = []
    means: dict[str, float] = {}
    for prog in sorted(programs.keys()):
        vals = programs[prog]
        s = summary(vals)
        means[prog] = s["mean"]
        print(f"{prog:<8} {int(s['count']):>4} {s['mean']:>7.2f} {s['median']:>7.2f} {s['std']:>7.2f}")
        by_prog_rows.append(
            {
                "program": prog,
                "count": str(int(s["count"])),
                "mean_final_score": f"{s['mean']:.4f}",
                "median_final_score": f"{s['median']:.4f}",
                "std_final_score": f"{s['std']:.4f}",
            }
        )

    # Save summaries for reuse.
    overall_out = base / "data" / "overall_summary.csv"
    by_prog_out = base / "data" / "summary_by_program.csv"
    with overall_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
        w.writeheader()
        w.writerows(overall_rows)
    with by_prog_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_prog_rows[0].keys()))
        w.writeheader()
        w.writerows(by_prog_rows)

    print("\nSaved:", overall_out)
    print("Saved:", by_prog_out)

    out_png = images_dir / "mean_final_by_program.png"
    try_plot_mean_by_program(means, out_png)
    if out_png.exists():
        print("Saved plot:", out_png)


if __name__ == "__main__":
    main()

