from __future__ import annotations

import math
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _ensure_dirs(lecture_dir: Path) -> tuple[Path, Path]:
    data_dir = lecture_dir / "data"
    img_dir = lecture_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, img_dir


def _mpl() -> tuple[object, object]:
    # Headless backend for reproducible image output.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    return matplotlib, plt


def run(lecture_type: str, lecture_dir: Path) -> None:
    """
    Generate a small dataset + one plot for in-class demo.

    Convention (per lecture folder):
    - data/dataset.csv (optional)
    - data/results.txt
    - images/demo.png
    """

    data_dir, img_dir = _ensure_dirs(lecture_dir)
    out_png = img_dir / "demo.png"
    out_txt = data_dir / "results.txt"

    if lecture_type == "sampling_ci":
        import numpy as np
        from scipy import stats

        _, plt = _mpl()

        # Small CI example (mean + t-interval).
        rng = np.random.default_rng(7)
        x = rng.normal(loc=50.0, scale=2.5, size=20)
        n = int(len(x))
        xbar = float(x.mean())
        s = float(x.std(ddof=1))
        alpha = 0.05
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        se = s / math.sqrt(n)
        ci = (xbar - tcrit * se, xbar + tcrit * se)

        # Sampling distribution of the sample mean.
        pop = rng.normal(loc=50.0, scale=2.5, size=20000)
        reps = 2000
        sample_n = 25
        means = rng.choice(pop, size=(reps, sample_n), replace=True).mean(axis=1)

        plt.figure(figsize=(6.2, 3.6))
        plt.hist(means, bins=35, color="#5b8bd0", edgecolor="white")
        plt.title("Sampling Distribution of Sample Mean (n=25)")
        plt.xlabel("sample mean")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(
            out_txt,
            "\n".join(
                [
                    f"n={n}",
                    f"mean={xbar:.4f}",
                    f"sd={s:.4f}",
                    f"95% CI=({ci[0]:.4f}, {ci[1]:.4f})",
                ]
            ),
        )
        return

    if lecture_type == "ttest":
        import numpy as np
        from scipy import stats

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        a = rng.normal(loc=60.0, scale=4.0, size=20)
        b = rng.normal(loc=64.0, scale=4.0, size=20)
        t, p = stats.ttest_ind(a, b, equal_var=False)

        plt.figure(figsize=(6.0, 3.6))
        plt.boxplot([a, b], labels=["A", "B"], patch_artist=True)
        plt.title("Two Independent Groups (t-test demo)")
        plt.ylabel("score")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(
            out_txt,
            "\n".join(
                [
                    f"nA={len(a)} meanA={a.mean():.3f} sdA={a.std(ddof=1):.3f}",
                    f"nB={len(b)} meanB={b.mean():.3f} sdB={b.std(ddof=1):.3f}",
                    f"Welch t={float(t):.4f} p={float(p):.6f}",
                ]
            ),
        )
        return

    if lecture_type == "paired_ttest":
        import numpy as np
        from scipy import stats

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        before = rng.normal(loc=60.0, scale=5.0, size=18)
        after = before + rng.normal(loc=2.0, scale=2.0, size=18)
        d = after - before

        t, p = stats.ttest_rel(after, before)
        d_mean = float(d.mean())
        d_sd = float(d.std(ddof=1))
        cohend = d_mean / d_sd if d_sd else float("nan")

        plt.figure(figsize=(6.2, 3.6))
        for i in range(len(before)):
            plt.plot([0, 1], [before[i], after[i]], marker="o", color="#5b8bd0", alpha=0.75)
        plt.xticks([0, 1], ["Before", "After"])
        plt.ylabel("score")
        plt.title("Paired Measurements (Before vs After)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(
            out_txt,
            "\n".join(
                [
                    f"n={len(d)} mean_diff={d_mean:.4f} sd_diff={d_sd:.4f}",
                    f"paired t={float(t):.4f} p={float(p):.6f}",
                    f"Cohen_d(paired)={cohend:.4f}",
                ]
            ),
        )
        return

    if lecture_type == "chi_square":
        import numpy as np
        from scipy.stats import chi2_contingency

        _, plt = _mpl()

        # 3x2 contingency table (counts).
        table = np.array([[18, 6], [12, 10], [16, 4]], dtype=float)
        chi2, p, dof, _expected = chi2_contingency(table)

        plt.figure(figsize=(6.0, 3.6))
        im = plt.imshow(table, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks([0, 1], ["Python", "R"])
        plt.yticks([0, 1, 2], ["CSE", "ECE", "AIML"])
        for (i, j), v in np.ndenumerate(table):
            plt.text(j, i, str(int(v)), ha="center", va="center", color="black", fontsize=9)
        plt.title("Contingency Table (Counts)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"chi2={float(chi2):.4f}\ndof={int(dof)}\np={float(p):.6f}\n")
        return

    if lecture_type == "anova":
        import numpy as np
        from scipy import stats

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        g1 = rng.normal(loc=60.0, scale=4.0, size=20)
        g2 = rng.normal(loc=66.0, scale=4.0, size=20)
        g3 = rng.normal(loc=72.0, scale=4.0, size=20)
        fstat, p = stats.f_oneway(g1, g2, g3)

        plt.figure(figsize=(6.0, 3.6))
        plt.boxplot([g1, g2, g3], labels=["G1", "G2", "G3"], patch_artist=True)
        plt.title("Three Groups (One-way ANOVA demo)")
        plt.ylabel("score")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"F={float(fstat):.4f}\np={float(p):.6f}\n")
        return

    if lecture_type == "nonparametric":
        import numpy as np
        from scipy import stats

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        # Skewed samples
        g1 = rng.lognormal(mean=1.0, sigma=0.6, size=25)
        g2 = rng.lognormal(mean=1.2, sigma=0.6, size=25)
        u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        plt.figure(figsize=(6.0, 3.6))
        plt.boxplot([g1, g2], labels=["G1", "G2"], patch_artist=True)
        plt.title("Skewed Groups (Mann-Whitney U demo)")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(
            out_txt,
            "\n".join(
                [
                    f"U={float(u):.4f}",
                    f"p={float(p):.6f}",
                    f"median(G1)={float(np.median(g1)):.4f}",
                    f"median(G2)={float(np.median(g2)):.4f}",
                ]
            ),
        )
        return

    if lecture_type == "case_interpretation":
        import numpy as np

        _, plt = _mpl()

        # Study summaries: (mean1-mean2)/pooled_sd
        studies = ["S1", "S2", "S3", "S4"]
        n1 = np.array([20, 30, 25, 18], dtype=float)
        m1 = np.array([72, 75, 65, 80], dtype=float)
        s1 = np.array([10, 12, 9, 8], dtype=float)
        n2 = np.array([20, 30, 25, 18], dtype=float)
        m2 = np.array([68, 70, 66, 76], dtype=float)
        s2 = np.array([10, 12, 9, 8], dtype=float)

        sp2 = ((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / (n1 + n2 - 2)
        sp = np.sqrt(sp2)
        d = (m1 - m2) / sp

        y = np.arange(len(studies))
        plt.figure(figsize=(6.2, 3.6))
        plt.axvline(0, color="black", linewidth=1)
        plt.scatter(d, y, color="#5b8bd0")
        plt.yticks(y, studies)
        for yi, di in zip(y, d.tolist()):
            plt.text(di + 0.02, yi, f"{di:.2f}", va="center", fontsize=9)
        plt.title("Effect Sizes (Cohen's d) by Study")
        plt.xlabel("Cohen's d")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        lines = [f"{s}: d={di:.4f}" for s, di in zip(studies, d.tolist())]
        _write_text(out_txt, "\n".join(lines))
        return

    if lecture_type == "corr_reg":
        import numpy as np

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        x = rng.uniform(0, 10, size=80)
        y = 3.0 * x + rng.normal(0, 6.0, size=80)
        r = float(np.corrcoef(x, y)[0, 1])

        plt.figure(figsize=(6.2, 3.6))
        plt.scatter(x, y, s=18, color="#5b8bd0", alpha=0.8)
        plt.title(f"Scatter Plot (r={r:.3f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"correlation_r={r:.6f}\n")
        return

    if lecture_type == "slr":
        import numpy as np

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        x = rng.uniform(0, 10, size=80)
        y = 10.0 + 2.5 * x + rng.normal(0, 4.0, size=80)

        b1, b0 = np.polyfit(x, y, deg=1)
        yhat = b0 + b1 * x
        resid = y - yhat

        fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6))
        ax[0].scatter(x, y, s=18, color="#5b8bd0", alpha=0.8)
        xs = np.linspace(x.min(), x.max(), 200)
        ax[0].plot(xs, b0 + b1 * xs, color="black", linewidth=2)
        ax[0].set_title("SLR Fit")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")

        ax[1].scatter(yhat, resid, s=18, color="#d07b5b", alpha=0.8)
        ax[1].axhline(0, color="black", linewidth=1)
        ax[1].set_title("Residuals vs Fitted")
        ax[1].set_xlabel("yhat")
        ax[1].set_ylabel("residual")

        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        _write_text(out_txt, f"b0={float(b0):.4f}\nb1={float(b1):.4f}\n")
        return

    if lecture_type == "mlr":
        import numpy as np
        from sklearn.linear_model import LinearRegression

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 120
        x1 = rng.normal(0, 1, size=n)
        x2 = rng.normal(0, 1, size=n)
        y = 2.0 + 1.5 * x1 - 0.7 * x2 + rng.normal(0, 0.8, size=n)

        X = np.column_stack([x1, x2])
        model = LinearRegression().fit(X, y)

        coefs = [float(model.intercept_), float(model.coef_[0]), float(model.coef_[1])]
        labels = ["intercept", "x1", "x2"]

        plt.figure(figsize=(6.2, 3.6))
        bars = plt.bar(labels, coefs, color=["#777777", "#5b8bd0", "#d07b5b"])
        for b, v in zip(bars, coefs):
            plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.title("MLR Coefficients (demo)")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"intercept={coefs[0]:.4f}\nb1={coefs[1]:.4f}\nb2={coefs[2]:.4f}\n")
        return

    if lecture_type == "poly_logit":
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve, roc_auc_score

        _, plt = _mpl()

        rng = np.random.default_rng(7)

        # Polynomial regression (visual)
        x = np.linspace(-3, 3, 120)
        y = 1.0 + 0.5 * x + 1.2 * (x**2) + rng.normal(0, 1.0, size=len(x))
        coeffs = np.polyfit(x, y, deg=2)
        yhat = np.polyval(coeffs, x)

        # Logistic regression (ROC)
        n = 250
        x1 = rng.normal(0, 1, size=n)
        x2 = rng.normal(0, 1, size=n)
        logit = 0.8 * x1 - 1.1 * x2 + 0.2
        p = 1 / (1 + np.exp(-logit))
        y_cls = (rng.uniform(0, 1, size=n) < p).astype(int)
        X = np.column_stack([x1, x2])
        clf = LogisticRegression(solver="lbfgs").fit(X, y_cls)
        prob = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_cls, prob)
        auc = float(roc_auc_score(y_cls, prob))

        fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6))
        ax[0].scatter(x, y, s=14, color="#5b8bd0", alpha=0.7)
        ax[0].plot(x, yhat, color="black", linewidth=2)
        ax[0].set_title("Polynomial Fit (degree 2)")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")

        ax[1].plot(fpr, tpr, color="#d07b5b", linewidth=2, label=f"AUC={auc:.3f}")
        ax[1].plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
        ax[1].set_title("ROC Curve (Logistic)")
        ax[1].set_xlabel("FPR")
        ax[1].set_ylabel("TPR")
        ax[1].legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        _write_text(out_txt, f"poly_coeffs(deg2)={coeffs.tolist()}\nroc_auc={auc:.6f}\n")
        return

    if lecture_type == "multicollinearity":
        import numpy as np

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 200
        x1 = rng.normal(0, 1, size=n)
        x2 = 0.95 * x1 + rng.normal(0, 0.15, size=n)
        x3 = rng.normal(0, 1, size=n)
        C = np.corrcoef(np.column_stack([x1, x2, x3]).T)

        plt.figure(figsize=(6.0, 3.6))
        im = plt.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        labels = ["x1", "x2", "x3"]
        plt.xticks(range(3), labels)
        plt.yticks(range(3), labels)
        for i in range(3):
            for j in range(3):
                plt.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", color="black", fontsize=9)
        plt.title("Correlation Heatmap (multicollinearity)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"corr(x1,x2)={float(C[0,1]):.6f}\n")
        return

    if lecture_type == "vif_regularization":
        import numpy as np
        from sklearn.linear_model import LinearRegression, Ridge, Lasso

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 250
        x1 = rng.normal(0, 1, size=n)
        x2 = 0.9 * x1 + rng.normal(0, 0.3, size=n)
        x3 = rng.normal(0, 1, size=n)
        y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(0, 1.0, size=n)

        X = np.column_stack([x1, x2, x3])
        feature_names = ["x1", "x2", "x3"]

        # VIF (basic via R^2 regression)
        vifs = []
        for j in range(X.shape[1]):
            yj = X[:, j]
            X_others = np.delete(X, j, axis=1)
            r2 = LinearRegression().fit(X_others, yj).score(X_others, yj)
            vif = 1.0 / (1.0 - r2) if r2 < 0.999999 else float("inf")
            vifs.append(float(vif))

        ols = LinearRegression().fit(X, y)
        ridge = Ridge(alpha=1.0).fit(X, y)
        lasso = Lasso(alpha=0.05, max_iter=10000).fit(X, y)

        coefs = {
            "OLS": [float(c) for c in ols.coef_],
            "Ridge": [float(c) for c in ridge.coef_],
            "Lasso": [float(c) for c in lasso.coef_],
        }

        # Coefficient bar plot
        plt.figure(figsize=(6.4, 3.8))
        x = np.arange(len(feature_names))
        w = 0.25
        plt.bar(x - w, coefs["OLS"], width=w, label="OLS")
        plt.bar(x, coefs["Ridge"], width=w, label="Ridge")
        plt.bar(x + w, coefs["Lasso"], width=w, label="Lasso")
        plt.xticks(x, feature_names)
        plt.axhline(0, color="black", linewidth=1)
        plt.title("Coefficients: OLS vs Ridge vs Lasso")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        lines = [f"VIF({n})={v:.4f}" for n, v in zip(feature_names, vifs)]
        lines.append(f"OLS_coef={coefs['OLS']}")
        lines.append(f"Ridge_coef={coefs['Ridge']}")
        lines.append(f"Lasso_coef={coefs['Lasso']}")
        _write_text(out_txt, "\n".join(lines))
        return

    if lecture_type == "cv_tuning":
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_score

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 220
        x1 = rng.normal(0, 1, size=n)
        x2 = rng.normal(0, 1, size=n)
        x3 = rng.normal(0, 1, size=n)
        y = 2.0 + 1.2 * x1 - 0.8 * x2 + 0.4 * x3 + rng.normal(0, 1.2, size=n)
        X = np.column_stack([x1, x2, x3])

        alphas = [0.01, 0.1, 1.0, 10.0, 50.0]
        cv = KFold(n_splits=5, shuffle=True, random_state=7)
        means = []
        for a in alphas:
            scores = cross_val_score(Ridge(alpha=a), X, y, cv=cv, scoring="neg_root_mean_squared_error")
            means.append(float(scores.mean()))

        best_idx = max(range(len(alphas)), key=lambda i: means[i])  # higher is better (less negative)
        best_alpha = alphas[best_idx]

        plt.figure(figsize=(6.2, 3.6))
        plt.plot(alphas, means, marker="o", color="#5b8bd0")
        plt.xscale("log")
        plt.title("CV Score vs Ridge alpha")
        plt.xlabel("alpha (log scale)")
        plt.ylabel("mean CV score (neg RMSE)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"alphas={alphas}\nmean_scores={means}\nbest_alpha={best_alpha}\n")
        return

    if lecture_type == "regression_case":
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 300
        x1 = rng.normal(0, 1, size=n)
        x2 = rng.normal(0, 1, size=n)
        x3 = rng.normal(0, 1, size=n)
        y = 5.0 + 2.0 * x1 - 1.0 * x2 + 0.7 * x3 + rng.normal(0, 1.5, size=n)
        X = np.column_stack([x1, x2, x3])

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
        model = LinearRegression().fit(Xtr, ytr)
        pred = model.predict(Xte)
        rmse = math.sqrt(float(mean_squared_error(yte, pred)))
        r2 = float(r2_score(yte, pred))

        plt.figure(figsize=(6.0, 3.8))
        plt.scatter(yte, pred, s=18, color="#5b8bd0", alpha=0.8)
        mn = min(float(yte.min()), float(pred.min()))
        mx = max(float(yte.max()), float(pred.max()))
        plt.plot([mn, mx], [mn, mx], color="black", linewidth=1)
        plt.title("Predicted vs Actual (Test Set)")
        plt.xlabel("actual y")
        plt.ylabel("predicted y")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"rmse={rmse:.4f}\nr2={r2:.4f}\n")
        return

    if lecture_type == "feature_intro":
        import numpy as np
        import pandas as pd

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 200
        user_id = rng.integers(1001, 1021, size=n)
        amount = rng.lognormal(mean=4.0, sigma=0.5, size=n)  # positive spending
        discount = rng.uniform(0.0, 0.2, size=n)
        net = amount * (1 - discount)
        df = pd.DataFrame({"user_id": user_id, "amount": amount, "discount": discount, "net_amount": net})
        (data_dir / "dataset.csv").write_text(df.to_csv(index=False), encoding="utf-8", newline="\n")

        plt.figure(figsize=(6.2, 3.6))
        plt.hist(net, bins=30, color="#5b8bd0", edgecolor="white")
        plt.title("Engineered Feature: net_amount distribution")
        plt.xlabel("net_amount")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"rows={len(df)}\nnet_amount_mean={float(df['net_amount'].mean()):.4f}\n")
        return

    if lecture_type == "feature_selection":
        import numpy as np
        import pandas as pd

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 240
        X = rng.normal(0, 1, size=(n, 6))
        # Make y depend more on a few features.
        y = 2.0 * X[:, 0] - 1.5 * X[:, 2] + 0.5 * X[:, 4] + rng.normal(0, 1.0, size=n)
        cols = [f"x{i+1}" for i in range(6)]
        df = pd.DataFrame(X, columns=cols)
        df["y"] = y
        (data_dir / "dataset.csv").write_text(df.to_csv(index=False), encoding="utf-8", newline="\n")

        corrs = [float(np.corrcoef(df[c].to_numpy(), y)[0, 1]) for c in cols]
        abs_corrs = [abs(c) for c in corrs]

        plt.figure(figsize=(6.2, 3.6))
        bars = plt.bar(cols, abs_corrs, color="#5b8bd0")
        for b, v in zip(bars, abs_corrs):
            plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.title("Filter-style score: |corr(feature, target)|")
        plt.ylabel("absolute correlation")
        plt.ylim(0, max(abs_corrs) * 1.2)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        ranked = sorted(zip(cols, corrs), key=lambda t: abs(t[1]), reverse=True)
        lines = ["Ranked features by |corr| (desc):"]
        lines.extend([f"{c}: corr={r:.4f}" for c, r in ranked])
        _write_text(out_txt, "\n".join(lines))
        return

    if lecture_type == "pca":
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 300
        # Correlated features
        z1 = rng.normal(0, 1, size=n)
        z2 = 0.7 * z1 + rng.normal(0, 1, size=n)
        z3 = -0.4 * z1 + 0.6 * z2 + rng.normal(0, 1, size=n)
        X = np.column_stack([z1, z2, z3])
        Xs = StandardScaler().fit_transform(X)

        pca = PCA().fit(Xs)
        evr = pca.explained_variance_ratio_

        plt.figure(figsize=(6.2, 3.6))
        xs = np.arange(1, len(evr) + 1)
        plt.bar(xs, evr, color="#5b8bd0")
        plt.plot(xs, np.cumsum(evr), color="black", marker="o")
        plt.xticks(xs, [f"PC{i}" for i in xs])
        plt.ylim(0, 1.05)
        plt.title("PCA Explained Variance (and cumulative)")
        plt.ylabel("ratio")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, "explained_variance_ratio=" + ",".join(f"{v:.6f}" for v in evr))
        return

    if lecture_type == "factor_lda":
        import numpy as np
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 300
        # 3-class dataset in 4D
        X0 = rng.normal(loc=[0, 0, 0, 0], scale=1.0, size=(n // 3, 4))
        X1 = rng.normal(loc=[2, 2, 0, 0], scale=1.0, size=(n // 3, 4))
        X2 = rng.normal(loc=[-2, 2, 1, -1], scale=1.0, size=(n - 2 * (n // 3), 4))
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2))

        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
        Z = lda.transform(X)

        plt.figure(figsize=(6.2, 3.8))
        for cls, color in [(0, "#5b8bd0"), (1, "#d07b5b"), (2, "#6aa84f")]:
            mask = y == cls
            plt.scatter(Z[mask, 0], Z[mask, 1], s=16, alpha=0.75, label=f"class {cls}", color=color)
        plt.title("LDA Projection (2D)")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, "LDA demo: 3 classes projected to 2D.\n")
        return

    if lecture_type == "kpca_tsne":
        import numpy as np
        from sklearn.datasets import make_moons
        from sklearn.manifold import TSNE

        _, plt = _mpl()

        X, y = make_moons(n_samples=350, noise=0.08, random_state=7)
        Z = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=7).fit_transform(X)

        plt.figure(figsize=(6.2, 3.8))
        plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="coolwarm", s=16, alpha=0.85)
        plt.title("t-SNE Visualization (two moons)")
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, "t-SNE demo on two-moons dataset.\n")
        return

    if lecture_type == "adv_feature_eng":
        import numpy as np
        import pandas as pd

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 300
        user_id = rng.integers(1001, 1031, size=n)
        amount = rng.lognormal(mean=4.0, sigma=0.6, size=n)
        day = rng.integers(1, 31, size=n)
        df = pd.DataFrame({"user_id": user_id, "day": day, "amount": amount})

        # Aggregation feature: total spend per user
        agg = df.groupby("user_id")["amount"].sum().reset_index(name="total_spend")
        (data_dir / "dataset.csv").write_text(df.to_csv(index=False), encoding="utf-8", newline="\n")
        (data_dir / "engineered.csv").write_text(agg.to_csv(index=False), encoding="utf-8", newline="\n")

        plt.figure(figsize=(6.2, 3.6))
        plt.hist(agg["total_spend"], bins=20, color="#5b8bd0", edgecolor="white")
        plt.title("Aggregation Feature: total_spend per user")
        plt.xlabel("total_spend")
        plt.ylabel("users")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"users={len(agg)}\nmean_total_spend={float(agg['total_spend'].mean()):.4f}\n")
        return

    if lecture_type == "pca_clustering":
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        # 3 clusters in 5D
        X0 = rng.normal(loc=0.0, scale=1.0, size=(120, 5))
        X1 = rng.normal(loc=3.0, scale=1.0, size=(120, 5))
        X2 = rng.normal(loc=[-3, 3, -2, 2, 0], scale=1.0, size=(120, 5))
        X = np.vstack([X0, X1, X2])

        Xs = StandardScaler().fit_transform(X)
        Z = PCA(n_components=2, random_state=7).fit_transform(Xs)
        labels = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(Z)

        plt.figure(figsize=(6.2, 3.8))
        plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=16, alpha=0.85)
        plt.title("PCA (2D) + KMeans Clusters")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, "kmeans_clusters=3 (in PCA space)\n")
        return

    if lecture_type == "ts_intro":
        import numpy as np
        import pandas as pd

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 180
        t = np.arange(n)
        trend = 0.05 * t
        season = 2.0 * np.sin(2 * np.pi * t / 30)
        noise = rng.normal(0, 0.8, size=n)
        y = 20 + trend + season + noise
        df = pd.DataFrame({"t": t, "y": y})
        (data_dir / "dataset.csv").write_text(df.to_csv(index=False), encoding="utf-8", newline="\n")

        plt.figure(figsize=(6.4, 3.6))
        plt.plot(t, y, color="#5b8bd0", linewidth=1.8)
        plt.title("Synthetic Time Series (trend + seasonality)")
        plt.xlabel("time")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"n={n}\n")
        return

    if lecture_type == "smoothing":
        import numpy as np

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 180
        t = np.arange(n)
        y = 20 + 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.9, size=n)

        # Moving average (window=7)
        w = 7
        ma = np.convolve(y, np.ones(w) / w, mode="valid")
        t_ma = t[w - 1 :]

        # Exponential smoothing
        alpha = 0.3
        es = np.zeros_like(y)
        es[0] = y[0]
        for i in range(1, n):
            es[i] = alpha * y[i] + (1 - alpha) * es[i - 1]

        plt.figure(figsize=(6.4, 3.6))
        plt.plot(t, y, color="#cccccc", label="original", linewidth=1)
        plt.plot(t_ma, ma, color="#5b8bd0", label="moving avg (7)", linewidth=2)
        plt.plot(t, es, color="#d07b5b", label=f"exp smooth (alpha={alpha})", linewidth=2)
        plt.title("Smoothing Techniques")
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"window={w}\nalpha={alpha}\n")
        return

    if lecture_type == "ar_ma":
        import numpy as np

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 250
        eps = rng.normal(0, 1, size=n)

        # AR(1)
        phi = 0.7
        ar = np.zeros(n)
        for i in range(1, n):
            ar[i] = phi * ar[i - 1] + eps[i]

        # MA(1)
        theta = 0.8
        ma = np.zeros(n)
        for i in range(1, n):
            ma[i] = eps[i] + theta * eps[i - 1]

        fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
        ax[0].plot(ar, color="#5b8bd0")
        ax[0].set_title(f"AR(1) phi={phi}")
        ax[1].plot(ma, color="#d07b5b")
        ax[1].set_title(f"MA(1) theta={theta}")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        _write_text(out_txt, f"phi={phi}\ntheta={theta}\n")
        return

    if lecture_type == "arima":
        import numpy as np
        from statsmodels.tsa.arima.model import ARIMA

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 220
        t = np.arange(n)
        y = 30 + 0.07 * t + rng.normal(0, 1.0, size=n)

        train_n = 180
        y_tr = y[:train_n]
        y_te = y[train_n:]

        model = ARIMA(y_tr, order=(1, 1, 1)).fit()
        fc = model.forecast(steps=len(y_te))

        plt.figure(figsize=(6.4, 3.6))
        plt.plot(t[:train_n], y_tr, label="train", color="#5b8bd0")
        plt.plot(t[train_n:], y_te, label="test", color="#cccccc")
        plt.plot(t[train_n:], fc, label="forecast", color="#d07b5b", linewidth=2)
        plt.title("ARIMA Forecast (demo)")
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, "model=ARIMA(1,1,1)\n")
        return

    if lecture_type == "stationarity":
        import numpy as np

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 200
        t = np.arange(n)
        y = 15 + 0.08 * t + rng.normal(0, 1.0, size=n)
        yd = np.diff(y, n=1)

        fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=False)
        ax[0].plot(t, y, color="#5b8bd0")
        ax[0].set_title("Original series (trend -> non-stationary mean)")
        ax[1].plot(np.arange(len(yd)), yd, color="#d07b5b")
        ax[1].set_title("First difference (often more stationary)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        _write_text(out_txt, "differencing=1\n")
        return

    if lecture_type == "adf":
        import numpy as np
        from statsmodels.tsa.stattools import adfuller

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 220
        t = np.arange(n)
        y = 10 + 0.06 * t + rng.normal(0, 1.0, size=n)
        yd = np.diff(y, n=1)

        p1 = float(adfuller(y, autolag="AIC")[1])
        p2 = float(adfuller(yd, autolag="AIC")[1])

        plt.figure(figsize=(6.4, 3.6))
        plt.plot(y, label="original", color="#5b8bd0")
        plt.plot(np.arange(1, n), yd, label="diff(1)", color="#d07b5b")
        plt.title("Original vs Differenced (ADF demo)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, f"ADF_p_original={p1:.6f}\nADF_p_diff1={p2:.6f}\n")
        return

    if lecture_type == "acf_pacf":
        import numpy as np
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 260
        # AR(1) series for illustration
        phi = 0.7
        eps = rng.normal(0, 1, size=n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i - 1] + eps[i]

        fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6))
        plot_acf(y, ax=ax[0], lags=30, alpha=0.05)
        ax[0].set_title("ACF")
        plot_pacf(y, ax=ax[1], lags=30, alpha=0.05, method="ywm")
        ax[1].set_title("PACF")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        _write_text(out_txt, f"series=AR(1) phi={phi}\n")
        return

    if lecture_type == "sarima":
        import numpy as np
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        _, plt = _mpl()

        rng = np.random.default_rng(7)
        n = 240
        t = np.arange(n)
        season = 2.5 * np.sin(2 * np.pi * t / 12)  # monthly seasonality
        trend = 0.03 * t
        y = 50 + trend + season + rng.normal(0, 0.8, size=n)

        train_n = 200
        y_tr = y[:train_n]
        y_te = y[train_n:]

        model = SARIMAX(y_tr, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        fc = model.forecast(steps=len(y_te))

        plt.figure(figsize=(6.4, 3.6))
        plt.plot(t[:train_n], y_tr, label="train", color="#5b8bd0")
        plt.plot(t[train_n:], y_te, label="test", color="#cccccc")
        plt.plot(t[train_n:], fc, label="forecast", color="#d07b5b", linewidth=2)
        plt.title("SARIMA Forecast (demo)")
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        _write_text(out_txt, "model=SARIMA(1,1,1)x(1,1,1,12)\n")
        return

    raise ValueError(f"Unknown lecture_type: {lecture_type}")
