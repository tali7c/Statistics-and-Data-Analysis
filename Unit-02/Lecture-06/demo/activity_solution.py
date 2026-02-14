from __future__ import annotations

import csv
import math
import statistics as stats
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


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


def pearson_r(xs: list[float], ys: list[float]) -> float:
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


def try_plot(outputs_dir: Path, xs: list[float], ys: list[float], out_png: Path, title: str, xlabel: str, ylabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except Exception:
        return

    outputs_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3.8))
    plt.scatter(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def try_plot_bar(means: dict[str, float], out_png: Path, title: str, ylabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except Exception:
        return

    programs = list(means.keys())
    ys = [means[p] for p in programs]
    plt.figure(figsize=(6.2, 3.6))
    bars = plt.bar(programs, ys)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, max(ys) * 1.15)
    for b, y in zip(bars, ys):
        plt.text(b.get_x() + b.get_width() / 2, y + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def try_plot_hist(values: list[float], out_png: Path, title: str, xlabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except Exception:
        return

    plt.figure(figsize=(6.2, 3.6))
    plt.hist(values, bins=8, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    base = Path(".")
    data_path = base / "data" / "activity_student_dataset.csv"
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(data_path)

    final_scores = [float(r["final_score"]) for r in rows]
    study_hours = [float(r["study_hours"]) for r in rows]
    sm_hours = [float(r["social_media_hours"]) for r in rows]
    attendance = [float(r["attendance_pct"]) for r in rows]

    mean_fs = stats.mean(final_scores)
    median_fs = stats.median(final_scores)
    modes = stats.multimode(final_scores)
    r_fs = max(final_scores) - min(final_scores)
    q1, q3 = quartiles_tukey(final_scores)
    iqr = q3 - q1
    var_s = stats.variance(final_scores)  # sample variance
    std_s = stats.stdev(final_scores)     # sample std dev

    r_hours = pearson_r(study_hours, final_scores)
    r_sm = pearson_r(sm_hours, final_scores)

    print("=== Activity Key Results ===")
    print(f"n = {len(final_scores)}")
    print(f"final_score: mean={mean_fs:.2f}  median={median_fs:.2f}  mode={modes}")
    print(f"final_score: range={r_fs:.2f}  Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
    print(f"final_score: sample variance={var_s:.2f}  sample std={std_s:.2f}")
    print(f"corr(study_hours, final_score) = {r_hours:.4f}")
    print(f"corr(social_media_hours, final_score) = {r_sm:.4f}")

    # Grouped summaries.
    by_prog: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        p = (r.get("program") or "").strip()
        by_prog.setdefault(p, {"final_score": [], "attendance_pct": []})
        by_prog[p]["final_score"].append(float(r["final_score"]))
        by_prog[p]["attendance_pct"].append(float(r["attendance_pct"]))

    prog_summary_rows: list[dict[str, str]] = []
    prog_means: dict[str, float] = {}
    print("\n=== By Program (final_score) ===")
    print(f"{'program':<6} {'n':>3} {'mean':>7} {'median':>7} {'std':>7}")
    print("-" * 34)
    for p in sorted(by_prog.keys()):
        vals = by_prog[p]["final_score"]
        n = len(vals)
        m = stats.mean(vals)
        med = stats.median(vals)
        sd = stats.stdev(vals) if n >= 2 else 0.0
        prog_means[p] = float(m)
        print(f"{p:<6} {n:>3} {m:>7.2f} {med:>7.2f} {sd:>7.2f}")
        prog_summary_rows.append(
            {
                "program": p,
                "count": str(n),
                "mean_final_score": f"{m:.4f}",
                "median_final_score": f"{med:.4f}",
                "std_final_score": f"{sd:.4f}",
                "mean_attendance_pct": f"{stats.mean(by_prog[p]['attendance_pct']):.4f}",
            }
        )

    # Save a couple of CSV outputs for reuse.
    out_prog = base / "data" / "summary_by_program.csv"
    with out_prog.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prog_summary_rows[0].keys()))
        w.writeheader()
        w.writerows(prog_summary_rows)
    print("\nSaved:", out_prog)

    out_overall = base / "data" / "overall_results.csv"
    with out_overall.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "mean_final_score",
                "median_final_score",
                "modes_final_score",
                "range_final_score",
                "q1_final_score",
                "q3_final_score",
                "iqr_final_score",
                "sample_variance_final_score",
                "sample_std_final_score",
                "corr_hours_final",
                "corr_social_media_final",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "n": str(len(final_scores)),
                "mean_final_score": f"{mean_fs:.4f}",
                "median_final_score": f"{median_fs:.4f}",
                "modes_final_score": ";".join(str(int(x)) for x in modes),
                "range_final_score": f"{r_fs:.4f}",
                "q1_final_score": f"{q1:.4f}",
                "q3_final_score": f"{q3:.4f}",
                "iqr_final_score": f"{iqr:.4f}",
                "sample_variance_final_score": f"{var_s:.4f}",
                "sample_std_final_score": f"{std_s:.4f}",
                "corr_hours_final": f"{r_hours:.6f}",
                "corr_social_media_final": f"{r_sm:.6f}",
            }
        )
    print("Saved:", out_overall)

    # Plots (optional).
    try_plot(images_dir, study_hours, final_scores, images_dir / "hours_vs_final_score.png",
             "Study Hours vs Final Score", "study_hours", "final_score")
    try_plot(images_dir, sm_hours, final_scores, images_dir / "social_media_vs_final_score.png",
             "Social Media Hours vs Final Score", "social_media_hours", "final_score")
    try_plot_bar(prog_means, images_dir / "mean_final_by_program.png",
                 "Mean Final Score by Program", "mean(final_score)")
    try_plot_hist(final_scores, images_dir / "final_score_hist.png", "Final Score (Histogram)", "final_score")

    print("\nSaved plots (if matplotlib is installed) into:", images_dir)


if __name__ == "__main__":
    main()

