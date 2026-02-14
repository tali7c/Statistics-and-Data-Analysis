from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
import re

REPO_URL = "https://github.com/tali7c/Statistics-and-Data-Analysis"
AUTHOR = "Tofik Ali"
INSTITUTE = "School of Computer Science, UPES Dehradun"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


_RE_NEEDS_ESCAPING = {
    "&": re.compile(r"(?<!\\)&"),
    "%": re.compile(r"(?<!\\)%"),
    "_": re.compile(r"(?<!\\)_"),
}


def _escape_latex_text_outside_math(s: str) -> str:
    """
    Escape a few LaTeX-special characters only outside $...$ math segments.
    This keeps common math like $H_0$ intact while fixing plain-text underscores like E_rc.
    """

    parts = s.split("$")
    for i in range(0, len(parts), 2):  # 0,2,4,... are outside math
        for ch, rx in _RE_NEEDS_ESCAPING.items():
            parts[i] = rx.sub(rf"\\{ch}", parts[i])
    return "$".join(parts)


def _frame_bullets(title: str, bullets: list[str], overlay: bool = True) -> str:
    opt = "[<+->]" if overlay else ""
    items = "\n".join(f"    \\item {_escape_latex_text_outside_math(b)}" for b in bullets)
    return textwrap.dedent(
        f"""
        \\begin{{frame}}{{{title}}}
          \\begin{{itemize}}{opt}
        {items}
          \\end{{itemize}}
        \\end{{frame}}
        """
    ).strip()


def _frame_math(title: str, eq: str, bullets: list[str] | None = None) -> str:
    if not bullets:
        return textwrap.dedent(
            f"""
            \\begin{{frame}}{{{title}}}
              {eq}
            \\end{{frame}}
            """
        ).strip()
    items = "\n".join(f"    \\item {_escape_latex_text_outside_math(b)}" for b in bullets)
    return textwrap.dedent(
        f"""
        \\begin{{frame}}{{{title}}}
          {eq}
          \\vspace{{0.6em}}
          \\begin{{itemize}}[<+->]
        {items}
          \\end{{itemize}}
        \\end{{frame}}
        """
    ).strip()


def _frame_ex(title: str, prompt: str) -> str:
    prompt = _escape_latex_text_outside_math(prompt)
    return textwrap.dedent(
        f"""
        \\begin{{frame}}{{{title}}}
          \\small
          {prompt}
        \\end{{frame}}
        """
    ).strip()


def _frame_sol(title: str, bullets: list[str]) -> str:
    items = "\n".join(f"    \\item {_escape_latex_text_outside_math(b)}" for b in bullets)
    return textwrap.dedent(
        f"""
        \\begin{{frame}}{{{title}}}
          \\begin{{itemize}}
        {items}
          \\end{{itemize}}
        \\end{{frame}}
        """
    ).strip()


def _slides_preamble(unit: int, lecture: int, subtitle: str, quick_links: list[tuple[str, str]]) -> str:
    u = f"{unit:02d}"
    l = f"{lecture:02d}"
    ql = "\n  ".join([f"\\hyperlink{{{lab}}}{{\\beamerbutton{{{txt}}}}}\\hspace{{0.6em}}" for txt, lab in quick_links])
    return textwrap.dedent(
        f"""
        \\documentclass{{beamer}}

        \\usetheme{{Berlin}}
        \\usecolortheme{{Orchid}}
        \\useoutertheme{{miniframes}}
        \\setbeamertemplate{{navigation symbols}}{{}}

        \\usepackage{{amsmath}}
        \\usepackage{{amssymb}}
        \\usepackage{{booktabs}}
        \\usepackage{{graphicx}}
        \\graphicspath{{{{../images/}}}}

        \\title[Statistics and Data Analysis]{{Statistics and Data Analysis}}
        \\subtitle{{Unit {u} -- Lecture {l}: {subtitle}}}
        \\author{{{AUTHOR}}}
        \\institute{{{INSTITUTE}}}
        \\date{{\\today}}

        \\begin{{document}}

        \\begin{{frame}}
          \\titlepage
          \\vspace{{-0.5em}}
          \\begin{{center}}
            \\small \\texttt{{{REPO_URL}}}
          \\end{{center}}
        \\end{{frame}}

        \\begin{{frame}}{{Quick Links}}
          \\centering
          {ql}
        \\end{{frame}}

        \\begin{{frame}}{{Agenda}}
          \\tableofcontents
        \\end{{frame}}
        """
    ).strip()


def _slides_end() -> str:
    return "\\end{document}\n"


def _notes_tex(unit: int, lecture: int, topic: str, outcomes: list[str], sections: list[tuple[str, list[str]]], exercises: list[tuple[str, str, list[str]]], exit_q: str) -> str:
    u = f"{unit:02d}"
    l = f"{lecture:02d}"
    out = [
        textwrap.dedent(
            f"""
            \\documentclass[11pt]{{article}}
            \\usepackage[utf8]{{inputenc}}
            \\usepackage[T1]{{fontenc}}
            \\usepackage{{geometry}}
            \\usepackage{{amsmath}}
            \\usepackage{{booktabs}}
            \\usepackage{{hyperref}}
            \\geometry{{margin=1in}}

            \\title{{Statistics and Data Analysis\\\\Unit {u} -- Lecture {l} Notes}}
            \\author{{{AUTHOR}}}
            \\date{{\\today}}

            \\begin{{document}}
            \\maketitle

            \\section*{{Topic}}
            {topic}

            \\subsection*{{Learning Outcomes}}
            \\begin{{itemize}}
            """
        ).strip()
    ]
    out.extend([f"  \\item {_escape_latex_text_outside_math(o)}" for o in outcomes])
    out.append("\\end{itemize}\n")

    out.append("\\section*{Detailed Notes}")
    out.append(
        textwrap.fill(
            "These notes are designed to be read alongside the slides. They expand each slide "
            "bullet into plain-language explanations, small worked examples, and common pitfalls. "
            "When a formula appears, emphasize (1) what each symbol means, (2) the assumptions "
            "needed to use it, and (3) how to interpret the final number in the problem context.",
            width=92,
        )
        + "\n"
    )

    for sec, pts in sections:
        out.append(f"\\section*{{{sec}}}")
        out.append("\\begin{itemize}")
        out.extend([f"  \\item {_escape_latex_text_outside_math(p)}" for p in pts])
        out.append("\\end{itemize}\n")

    out.append("\\section*{Exercises (with Solutions)}")
    for i, (title, prompt, sol) in enumerate(exercises, start=1):
        out.append(f"\\subsection*{{Exercise {i}: {title}}}")
        out.append(_escape_latex_text_outside_math(prompt))
        out.append("\\subsection*{Solution}")
        out.append("\\begin{itemize}")
        out.extend([f"  \\item {_escape_latex_text_outside_math(b)}" for b in sol])
        out.append("\\end{itemize}\n")

    out.append("\\section*{Exit Question}")
    out.append(_escape_latex_text_outside_math(exit_q) + "\n")

    out.append("\\section*{Demo (Python)}")
    out.append(
        textwrap.dedent(
            """
            Run from the lecture folder:
            \\begin{verbatim}
            python demo/demo.py
            \\end{verbatim}

            Output files:
            \\begin{itemize}
              \\item \\texttt{images/demo.png}
              \\item \\texttt{data/results.txt}
            \\end{itemize}
            """
        ).strip()
        + "\n"
    )

    out.append("\\section*{References}")
    out.append("\\begin{itemize}")
    out.append("  \\item Montgomery, D. C., \\& Runger, G. C. \\textit{Applied Statistics and Probability for Engineers}, Wiley.")
    out.append("  \\item Devore, J. L. \\textit{Probability and Statistics for Engineering and the Sciences}, Cengage.")
    out.append("  \\item McKinney, W. \\textit{Python for Data Analysis}, O'Reilly.")
    out.append("\\end{itemize}")
    out.append("\\end{document}\n")
    return "\n".join(out)


@dataclass(frozen=True)
class Lecture:
    unit: int
    lecture: int
    subtitle: str
    topic: str
    demo_type: str
    outcomes: list[str]
    sections: list[tuple[str, str, list[str], str | None]]  # (name, label, bullets, eq)
    exercises: list[tuple[str, str, list[str]]]  # (title, prompt, solution bullets)
    exit_question: str

    @property
    def dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / f"Unit-{self.unit:02d}" / f"Lecture-{self.lecture:02d}"


TYPE_LIBRARY: dict[str, dict[str, object]] = {
    "sampling_ci": {
        "outcomes": [
            "Differentiate population parameters and sample statistics",
            "Explain sampling bias vs random error",
            "Describe common sampling methods (SRS, stratified, cluster)",
            "Compute and interpret a basic confidence interval for a mean",
        ],
        "sections": [
            ("Sampling", "sec:sampling", ["Population vs sample", "Bias vs random error", "Representative sampling matters"], None),
            (
                "Confidence Intervals",
                "sec:ci",
                ["Interpretation: long-run coverage", "Width depends on n and variability", "CIs support decision-making with uncertainty"],
                "\\[ \\bar{x} \\pm t_{\\alpha/2,\\,n-1}\\,\\frac{s}{\\sqrt{n}} \\]",
            ),
        ],
        "exercises": [
            ("Parameter vs Statistic", "Give one example of a parameter and one of a statistic.", ["Parameter: population mean", "Statistic: sample mean"]),
            ("CI Interpretation", "In one sentence, what does a 95\\% CI mean (correctly)?", ["About 95\\% of such intervals contain the true mean in repeated sampling."]),
            ("Bias Scenario", "Why does convenience sampling create bias?", ["Because some groups are over/under-represented systematically."]),
        ],
        "exit": "If your CI is too wide, what two actions reduce its width (without cheating)?",
    },
    "ttest": {
        "outcomes": [
            "Define null and alternative hypotheses clearly",
            "Compute a one-sample t statistic (given summary)",
            "Explain p-value and significance level alpha",
            "Distinguish one-tailed vs two-tailed tests",
            "State key assumptions behind the t-test",
        ],
        "sections": [
            (
                "t-test Basics",
                "sec:ttest",
                ["H0/H1 setup", "Test statistic measures how far the sample is from H0", "Assumptions: independence, outliers, normality/CLT"],
                "\\[ t = \\frac{\\bar{x}-\\mu_0}{s/\\sqrt{n}},\\quad \\mathrm{df}=n-1 \\]",
            ),
            (
                "p-values",
                "sec:pvalues",
                ["p-value = probability of data (or more extreme) assuming H0", "Small p-value -> evidence against H0", "p-value is not effect size"],
                None,
            ),
        ],
        "exercises": [
            ("Write hypotheses", "Claim: mean score is 60. Write H0 and H1 for a two-sided test.", ["H0: mu = 60", "H1: mu != 60"]),
            ("Compute t", "Given n=25, xbar=53, s=10, test H0: mu=50. Compute t.", ["SE = 10/sqrt(25) = 2", "t = (53-50)/2 = 1.5", "df = 24"]),
            ("Tail choice", "You want to show a new method increases mean score. One-tailed or two-tailed?", ["One-tailed (right): H1: mu > mu0"]),
        ],
        "exit": "Why can a very small p-value still be unimportant in practice?",
    },
    "paired_ttest": {
        "outcomes": [
            "Differentiate paired vs independent designs",
            "Compute within-pair differences di",
            "Run a paired t-test (conceptually)",
            "Explain effect size and why we report it",
            "Interpret results in context (not only p-value)",
        ],
        "sections": [
            (
                "Paired Design",
                "sec:paired",
                ["Same unit measured twice (before/after)", "Analyze differences di = after - before", "Pairing reduces noise from individual differences"],
                "\\[ t = \\frac{\\bar{d}}{s_d/\\sqrt{n}},\\quad \\mathrm{df}=n-1 \\]",
            ),
            ("Effect Size", "sec:effect", ["p-value answers: evidence?", "Effect size answers: how big?", "Large n can make tiny effects significant"], "\\[ d = \\frac{\\bar{x}_1-\\bar{x}_2}{s_{\\mathrm{pooled}}} \\]"),
        ],
        "exercises": [
            ("Compute differences", "Before/After: (10,12), (12,12), (11,14), (9,10). Compute di and dbar.", ["di: 2,0,3,1", "dbar = 1.5"]),
            ("CI idea", "If the 95% CI for mean difference excludes 0, what does it suggest?", ["Evidence of a change (difference likely non-zero).", "Check magnitude and context."]),
            ("Interpret d", "If Cohen's d=0.3, what does it suggest (rule of thumb)?", ["Small effect (context dependent).", "Still may matter if cheap/safe to adopt."]),
        ],
        "exit": "Why can paired designs be more powerful than independent designs?",
    },
    "chi_square": {
        "outcomes": [
            "Explain when chi-square tests are used (counts/frequencies)",
            "Compute expected counts for a contingency table",
            "Compute chi-square statistic (basic)",
            "State assumptions (expected counts not too small)",
            "Interpret independence vs association",
        ],
        "sections": [
            (
                "Chi-square Tests",
                "sec:chi2",
                ["GOF: one categorical variable", "Independence: two categorical variables", "Compare observed O to expected E under H0"],
                "\\[ \\chi^2 = \\sum \\frac{(O-E)^2}{E} \\]",
            ),
            ("Expected Counts", "sec:expected", ["E_rc = (row total)(col total)/N", "df = (R-1)(C-1) for independence"], None),
        ],
        "exercises": [
            ("Expected counts", "Row totals 60/40, column totals 70/30, N=100. Compute E11.", ["E11 = 60*70/100 = 42"]),
            ("Interpret reject", "If you reject H0 in independence test, what do you conclude?", ["Evidence of association between variables.", "Not direction; not causation."]),
            ("Assumption", "Why do we worry about small expected counts?", ["Chi-square approximation can break when E is very small."]),
        ],
        "exit": "Why do chi-square tests use expected counts instead of only raw percentages?",
    },
    "anova": {
        "outcomes": [
            "Explain why ANOVA is used for comparing 3+ means",
            "Describe between-group vs within-group variation",
            "Interpret F statistic at a high level",
            "State main assumptions of one-way ANOVA",
            "Explain what a post-hoc test is",
        ],
        "sections": [
            ("ANOVA Concept", "sec:anova", ["One global test for equality of means", "Avoids inflating Type I error vs many t-tests", "If significant, follow with post-hoc"], "\\[ F = \\frac{\\text{between-group variation}}{\\text{within-group variation}} \\]"),
            ("Assumptions", "sec:assum", ["Independent observations", "Rough normality within groups (or robust with n)", "Similar variances across groups"], None),
        ],
        "exercises": [
            ("Write H0", "Compare 3 group means. What is H0?", ["H0: mu1 = mu2 = mu3"]),
            ("Within variance", "If within-group variance increases, what happens to F (all else equal)?", ["F tends to decrease; harder to detect differences."]),
            ("Next step", "ANOVA p-value is 0.01 at alpha=0.05. What next?", ["Reject H0.", "Run post-hoc to find which pairs differ."]),
        ],
        "exit": "Why are several pairwise t-tests not equivalent to one ANOVA?",
    },
    "nonparametric": {
        "outcomes": [
            "Explain why non-parametric tests are used",
            "Choose Mann-Whitney / Wilcoxon / Kruskal-Wallis",
            "Interpret p-values carefully",
            "Discuss statistical vs practical significance",
            "Explain multiple testing risk",
        ],
        "sections": [
            ("When to Use", "sec:when", ["Skewed data/outliers", "Ordinal scales", "Small sample and doubtful normality"], None),
            ("Common Tests", "sec:tests", ["Two independent groups: Mann-Whitney U", "Paired samples: Wilcoxon signed-rank", "3+ groups: Kruskal-Wallis"], None),
        ],
        "exercises": [
            ("Choose test", "Same students before/after training (skewed). Which test?", ["Wilcoxon signed-rank"]),
            ("Practical vs statistical", "Very small p-value but tiny difference: what should you report?", ["Report effect size and context; significance != importance."]),
            ("Multiple testing", "20 tests at alpha=0.05: expected false positives?", ["About 1 on average."]),
        ],
        "exit": "Give one reason to prefer a rank-based test over a mean-based test.",
    },
    "case_interpretation": {
        "outcomes": [
            "Interpret p-values and confidence intervals correctly",
            "Compute a simple effect size from summary statistics",
            "Identify common red flags (only p-values, many tests, no effect size)",
            "Write a cautious conclusion in plain language",
            "Avoid correlation-causation confusion",
        ],
        "sections": [
            ("Reading Results", "sec:read", ["Check n, center, spread", "Prefer CI + effect size", "Ask: what does it mean in the real world?"], None),
            ("Pitfalls", "sec:pitfalls", ["Multiple comparisons", "Selective reporting (p-hacking)", "Over-claiming causation"], None),
        ],
        "exercises": [
            ("Interpret CI", "95% CI for (new-old) is (1.2, 3.8). What does it suggest?", ["Likely positive effect (CI above 0).", "Magnitude between 1.2 and 3.8 units."]),
            ("Compute d", "A: n=20 mean=72 SD=10; B: n=20 mean=68 SD=10. Compute Cohen's d.", ["Pooled SD=10", "d=(72-68)/10=0.4"]),
            ("Cautious conclusion", "p-value=0.03 but effect size is tiny. What should you conclude?", ["Evidence of difference, but small magnitude.", "May not justify action without cost/benefit."]),
        ],
        "exit": "What is one red flag when a paper reports only p-values and no effect sizes?",
    },
    "corr_reg": {
        "outcomes": [
            "Differentiate correlation and regression",
            "Explain why correlation does not imply causation",
            "Interpret a scatter plot (trend, outliers)",
            "Define residual and why residuals matter",
        ],
        "sections": [
            ("Concepts", "sec:concepts", ["Correlation measures linear association", "Regression models Y as a function of X", "Regression has roles: predictors vs response"], None),
            ("Causation Warning", "sec:causation", ["Confounding can create misleading correlation", "Reverse causality is possible", "Causal claims need design or strong assumptions"], None),
        ],
        "exercises": [
            ("Pick response variable", "Predict house price using size and location. What is the response variable?", ["House price is the response (Y)."]),
            ("Interpret r", "If r=0.7 between study hours and score, what does it mean?", ["Strong positive linear association.", "Not proof of causation."]),
            ("Residual sign", "If y=74 and yhat=80, what is residual?", ["Residual = y - yhat = -6 (over-prediction)."]),
        ],
        "exit": "Give one example of a confounder that can create a misleading correlation.",
    },
    "slr": {
        "outcomes": [
            "Write the simple linear regression model",
            "Interpret slope and intercept in context",
            "Compute a prediction and a residual",
            "Explain R-squared (intuition)",
        ],
        "sections": [
            ("Model", "sec:model", ["y = b0 + b1 x + error", "Slope: expected change in y for 1-unit increase in x", "Intercept: predicted y at x=0 (interpret carefully)"], "\\[ y = \\beta_0 + \\beta_1 x + \\epsilon \\]"),
            ("Fit and Diagnostics", "sec:diag", ["Look at residual plots for patterns", "Outliers can dominate the fitted line", "High $R^2$ does not guarantee a good model"], None),
        ],
        "exercises": [
            ("Prediction", "Model: yhat = 10 + 2x. Predict y when x=7.", ["yhat = 24"]),
            ("Residual", "If actual y=20 at x=7, compute residual.", ["20-24 = -4"]),
            ("Interpret slope", "Slope is 5 thousand INR per extra room. Interpret.", ["Each extra room increases predicted price by ~5k INR (on average)."]),
        ],
        "exit": "Why do we check residual plots even if $R^2$ is high?",
    },
    "mlr": {
        "outcomes": [
            "Write the multiple linear regression model",
            "Interpret a coefficient as a partial effect",
            "Explain dummy variables for categories (basic)",
            "Explain adjusted R-squared (intuition)",
        ],
        "sections": [
            ("Model", "sec:model", ["y = b0 + b1 x1 + b2 x2 + ...", "Each coefficient is a partial effect (others fixed)", "Scaling helps when using regularization"], "\\[ y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\epsilon \\]"),
            ("Interpretation", "sec:interp", ["Dummy variables encode categories", "Adjusted $R^2$ penalizes unnecessary predictors", "Multicollinearity can harm interpretability"], None),
        ],
        "exercises": [
            ("Partial effect", "Model: yhat=5 + 0.8x1 + 2.0x2. Interpret coefficient 2.0.", ["Holding x1 fixed, +1 in x2 increases yhat by 2.0 units."]),
            ("Dummy variable", "Urban=1, Rural=0. If coef(Urban)=10, interpret.", ["Urban has predicted y about 10 units higher than Rural (all else equal)."]),
            ("Adjusted $R^2$", "Why use adjusted $R^2$ when comparing models with different number of predictors?", ["Because $R^2$ never decreases, adjusted $R^2$ penalizes extra predictors."]),
        ],
        "exit": "Why does adding a useless feature still increase (or keep) $R^2$?",
    },
    "poly_logit": {
        "outcomes": [
            "Explain polynomial features for modeling curvature",
            "Recognize overfitting risk with high degree",
            "Write logistic regression probability model (sigmoid)",
            "Compute precision and recall from a confusion matrix",
        ],
        "sections": [
            ("Polynomial Regression", "sec:poly", ["Add features $x, x^2, x^3, \\dots$", "Still linear in parameters", "Choose degree using validation"], None),
            ("Logistic Regression", "sec:logit", ["Outputs probability in (0,1)", "Threshold converts probability to class label", "Evaluate using confusion matrix / ROC"], "\\[ P(y=1\\mid x)=\\frac{1}{1+e^{-(\\beta_0+\\beta^T x)}} \\]"),
        ],
        "exercises": [
            ("Polynomial features", "For degree-2 polynomial, what features do we use from $x$?", ["Use $1, x, x^2$ (intercept + linear + quadratic)."]),
            ("Precision/recall", "TP=30 FP=10 FN=20 TN=40. Compute precision and recall.", ["Precision=30/(30+10)=0.75", "Recall=30/(30+20)=0.60"]),
            ("Threshold effect", "If threshold increases from 0.5 to 0.8, what tends to happen to precision and recall?", ["Precision often increases, recall often decreases."]),
        ],
        "exit": "Why is ROC curve useful when classes are imbalanced?",
    },
    "multicollinearity": {
        "outcomes": [
            "Define multicollinearity (high correlation among predictors)",
            "Explain why it harms interpretation (unstable coefficients)",
            "Recognize symptoms (large SEs, unstable signs)",
            "List common fixes (drop/combine/regularize)",
        ],
        "sections": [
            ("What and Why", "sec:why", ["Predictors overlap in information", "Coefficients become unstable", "Prediction may still be OK but interpretation suffers"], None),
            ("Detection", "sec:detect", ["Correlation matrix/heatmap (screening)", "VIF (next)", "Condition number (advanced)"], None),
        ],
        "exercises": [
            ("Identify", "If corr(x1,x2)=0.98, what risk do you expect?", ["High multicollinearity; unstable coefficients."]),
            ("Fix", "Name one fix for multicollinearity.", ["Drop one feature, combine features, or use ridge/PCA."]),
            ("Prediction vs interpretation", "Can multicollinearity still allow good prediction?", ["Yes, but individual coefficients are unreliable."]),
        ],
        "exit": "What does multicollinearity break first: prediction or interpretation (and why)?",
    },
    "vif_regularization": {
        "outcomes": [
            "Compute and interpret VIF (basic)",
            "Explain AIC/BIC as model selection criteria (intuition)",
            "Write ridge and lasso objectives",
            "Explain coefficient shrinkage and feature selection idea",
        ],
        "sections": [
            ("VIF", "sec:vif", ["Definition: $\\mathrm{VIF}_j = 1/(1-R_j^2)$", "Higher VIF -> more multicollinearity", "Rule of thumb thresholds (5/10)"], "\\[ \\mathrm{VIF}_j = \\frac{1}{1-R_j^2} \\]"),
            ("Ridge/Lasso", "sec:reg", ["Ridge uses L2 penalty (shrinks)", "Lasso uses L1 penalty (can set some to 0)", "Scale features before regularization"], "\\[ \\min \\sum (y-\\hat{y})^2 + \\lambda \\sum \\beta_j^2 \\quad \\text{(ridge)} \\]"),
        ],
        "exercises": [
            ("Compute VIF", "If $R_j^2=0.9$, compute $\\mathrm{VIF}_j$.", ["$\\mathrm{VIF}_j = 1/(1-0.9)=10$ (high)."]),
            ("Ridge vs lasso", "Which can produce exact zero coefficients?", ["Lasso (L1) can set some coefficients to 0."]),
            ("AIC/BIC meaning", "Lower AIC/BIC means what (conceptually)?", ["Better trade-off between fit and complexity (relative)."]),
        ],
        "exit": "Why can ridge help when predictors are highly correlated?",
    },
    "cv_tuning": {
        "outcomes": [
            "Explain train/validation/test split roles",
            "Describe k-fold cross-validation",
            "Explain grid search vs random search",
            "Avoid data leakage using pipelines",
        ],
        "sections": [
            ("Cross-validation", "sec:cv", ["CV estimates generalization more stably than one split", "k-fold repeats train/validate across folds", "Average score guides selection"], None),
            ("Hyper-parameter Tuning", "sec:tuning", ["Grid search tries all combos", "Random search samples combos efficiently", "Never tune on the test set"], None),
        ],
        "exercises": [
            ("Grid size", "3 parameters with 4 values each: how many combinations?", ["$4^3 = 64$"]),
            ("Leakage", "Is scaling on full dataset before split leakage?", ["Yes; fit preprocessing on training only."]),
            ("Why CV", "Why is a single train-test split misleading sometimes?", ["Performance depends on split; CV reduces variance."]),
        ],
        "exit": "Why must you never use the test set to choose hyperparameters?",
    },
    "regression_case": {
        "outcomes": [
            "Describe an end-to-end regression workflow",
            "Choose appropriate regression metrics (RMSE and $R^2$)",
            "Check overfitting (train vs test gap)",
            "Communicate results with plots (predicted vs actual, residuals)",
        ],
        "sections": [
            ("Workflow", "sec:workflow", ["Define target and inputs", "Prepare data and split chronologically if needed", "Fit baseline then iterate"], None),
            ("Evaluation", "sec:eval", ["Use RMSE/MSE/MAE and $R^2$", "Use plots: predicted vs actual, residuals", "Document limitations"], None),
        ],
        "exercises": [
            ("Metric choice", "Target is continuous (price). Should you use accuracy?", ["No; accuracy is for classification."]),
            ("Overfitting sign", "Train RMSE=5, test RMSE=20. What does it suggest?", ["Overfitting; try simpler model or regularization."]),
            ("Communication", "Name one plot to communicate regression quality.", ["Predicted vs actual scatter; residual plot."]),
        ],
        "exit": "What would you do first if the case study model performs poorly on the test set?",
    },
    "feature_intro": {
        "outcomes": [
            "Differentiate feature selection vs dimensionality reduction",
            "Explain why too many features can hurt (overfitting, cost)",
            "Describe a simple feature engineering pipeline",
            "Identify target leakage in engineered features",
        ],
        "sections": [
            ("Why Features", "sec:why", ["Features are how models see data", "Goal: represent signal and reduce noise", "Bad features -> bad models"], None),
            ("Selection vs Reduction", "sec:sel", ["Selection keeps a subset of original features", "Reduction creates new components (e.g., PCA)", "Validate choices using CV"], None),
        ],
        "exercises": [
            ("Selection or reduction", "Dropping 30 out of 100 features is selection or reduction?", ["Feature selection (subset)."]),
            ("Leakage", "Is using final exam score to predict final grade leakage?", ["Yes; it contains future/target information."]),
            ("Engineering example", "Give one time-based engineered feature.", ["Day-of-week, month, time-since-last-event, rolling average, etc."]),
        ],
        "exit": "Why can adding more features sometimes reduce test accuracy?",
    },
    "feature_selection": {
        "outcomes": [
            "Explain filter methods (variance, correlation, mutual information)",
            "Explain wrapper methods (RFE) at a high level",
            "Explain embedded methods (lasso, tree importance) at a high level",
            "Discuss pros/cons of each approach",
        ],
        "sections": [
            ("Filter Methods", "sec:filter", ["Fast scoring without training many models", "Examples: variance threshold, correlation with target", "May miss interactions"], None),
            ("Wrapper/Embedded", "sec:wrap", ["Wrapper: search subsets using a model (slow)", "Embedded: selection during training (lasso, trees)"], None),
        ],
        "exercises": [
            ("Low variance", "If a feature is almost constant, keep it?", ["Usually no; low variance adds little information."]),
            ("Redundant features", "Two features have corr=0.99. What might you do?", ["Drop one or use regularization/PCA."]),
            ("Wrapper trade-off", "Why is RFE slower than filters?", ["It trains many models on many subsets."]),
        ],
        "exit": "When would you prefer a fast filter method over a wrapper method?",
    },
    "pca": {
        "outcomes": [
            "Explain PCA as a variance-maximizing linear projection",
            "State why scaling is important before PCA",
            "Interpret explained variance ratio and scree plot",
            "Use PCA for visualization and noise reduction",
        ],
        "sections": [
            ("PCA Intuition", "sec:intuition", ["Find new axes (components) capturing maximum variance", "Components are orthogonal", "PC1 captures most variance"], None),
            ("Explained Variance", "sec:var", ["Explained variance ratio per component", "Choose k via scree plot / cumulative variance target", "Validate downstream performance"], None),
        ],
        "exercises": [
            ("Scaling", "Why scale features before PCA?", ["To prevent large-unit features dominating variance."]),
            ("Components", "Are PCA components original features?", ["No; they are linear combinations."]),
            ("Choosing k", "If first 2 PCs explain 88% and you need 90%, what do you do?", ["Add next PC(s) until target reached."]),
        ],
        "exit": "Why might PCA improve a model even though it discards some variance?",
    },
    "factor_lda": {
        "outcomes": [
            "Explain factor analysis as latent-factor modeling (intuition)",
            "Differentiate PCA vs factor analysis (goal/assumptions)",
            "Explain LDA as supervised dimensionality reduction/classifier",
            "Interpret a 2D LDA projection",
        ],
        "sections": [
            ("Factor Analysis", "sec:factor", ["Observed variables driven by a few latent factors", "Goal: explain correlations via factors", "Used for surveys/constructs"], None),
            ("LDA", "sec:lda", ["Supervised: uses labels", "Finds projection maximizing class separation", "Can classify and visualize"], None),
        ],
        "exercises": [
            ("Supervised?", "Is PCA supervised? Is LDA supervised?", ["PCA is unsupervised; LDA is supervised."]),
            ("Goal", "What does PCA optimize vs LDA (intuition)?", ["PCA: variance captured; LDA: class separability."]),
            ("Use case", "Labeled A/B/C data, want 2D plot separating classes. PCA or LDA?", ["LDA (uses labels for separation)."]),
        ],
        "exit": "Why can LDA separate classes better than PCA on labeled data?",
    },
    "kpca_tsne": {
        "outcomes": [
            "Explain why nonlinear methods are sometimes needed",
            "Describe kernel PCA idea (high level)",
            "Describe t-SNE purpose (visualization) and pitfalls",
            "Choose PCA vs t-SNE appropriately",
        ],
        "sections": [
            ("Kernel PCA", "sec:kpca", ["Implicitly map to higher-dimensional space via kernel", "Apply PCA in that space", "Captures nonlinear structure"], None),
            ("t-SNE", "sec:tsne", ["Mainly for 2D/3D visualization", "Preserves local neighborhoods", "Global distances can be misleading"], None),
        ],
        "exercises": [
            ("Use case", "Name one warning when interpreting t-SNE plots.", ["Global distances and cluster sizes can be misleading."]),
            ("Randomness", "What should you do if t-SNE changes across runs?", ["Set seed and check stability."]),
            ("Kernel PCA benefit", "Why kernel PCA can help on circular data?", ["It can capture nonlinear manifold structure."]),
        ],
        "exit": "Why should we avoid using t-SNE coordinates directly as model features (usually)?",
    },
    "adv_feature_eng": {
        "outcomes": [
            "Create interaction features when meaningful",
            "Create aggregation features from transactional data",
            "Engineer time-based features (lags/rolling)",
            "Avoid leakage and look-ahead bias",
        ],
        "sections": [
            ("Interactions", "sec:inter", ["Products and ratios capture combined effects", "Use domain knowledge", "Validate with CV"], None),
            ("Aggregations", "sec:agg", ["Per-user totals/means/counts", "Rolling windows (last 7/30 days)", "Avoid using future data"], None),
        ],
        "exercises": [
            ("Interaction", "Give one interaction feature for house price.", ["size_m2 * location_score (example)."]),
            ("Aggregation", "Name one per-user aggregation for churn prediction.", ["days_since_last_purchase (example)."]),
            ("Leakage", "Is using next-30-days spend to predict churn today leakage?", ["Yes; it uses future info."]),
        ],
        "exit": "How does cross-validation help detect whether engineered features overfit?",
    },
    "pca_clustering": {
        "outcomes": [
            "Run PCA before clustering for visualization/stability",
            "Explain why scaling matters for clustering",
            "Use KMeans and interpret clusters cautiously",
            "Visualize clusters in PCA space",
        ],
        "sections": [
            ("Pipeline", "sec:pipeline", ["Scale features", "Run PCA (2D for visualization)", "Cluster (KMeans) and visualize"], None),
            ("Interpretation", "sec:interpret", ["Clusters are patterns, not truth", "Check stability across seeds/k", "Explain clusters using original variables"], None),
        ],
        "exercises": [
            ("Scaling", "Why scale before KMeans?", ["Distance-based; scale dominates otherwise."]),
            ("Choose k", "Name one heuristic to choose k.", ["Elbow, silhouette, domain knowledge."]),
            ("Explain cluster", "How to explain cluster to non-technical audience?", ["Describe in original variables (high spend, frequent visits, etc.)."]),
        ],
        "exit": "Why should you validate cluster stability before using clusters for decisions?",
    },
    "ts_intro": {
        "outcomes": [
            "Define time series and why order matters",
            "Identify trend, seasonality, and noise",
            "Explain autocorrelation (intuition)",
            "Explain why random shuffling breaks time series analysis",
        ],
        "sections": [
            ("Components", "sec:comp", ["Trend: long-term movement", "Seasonality: repeating pattern", "Noise: irregular fluctuations"], None),
            ("Autocorrelation", "sec:auto", ["Correlation with past values", "Important for AR/MA/ARIMA models", "Shows persistence of shocks"], None),
        ],
        "exercises": [
            ("Order matters", "Why should train/test split be chronological for time series?", ["To avoid future-to-past leakage."]),
            ("Seasonality example", "Give one seasonal pattern in campus data.", ["Weekly cafe sales (weekday vs weekend), etc."]),
            ("Autocorr meaning", "If lag-1 autocorrelation is strong positive, what does it suggest?", ["Values tend to persist from one step to next."]),
        ],
        "exit": "In one sentence: what is seasonality and why does it matter for forecasting?",
    },
    "smoothing": {
        "outcomes": [
            "Explain why smoothing is used (noise reduction)",
            "Describe moving average and its window effect",
            "Describe exponential smoothing and alpha effect",
            "Discuss responsiveness vs smoothness trade-off",
        ],
        "sections": [
            ("Moving Average", "sec:ma", ["Average last k points", "Larger k -> smoother but more lag", "Good for trend visualization"], None),
            ("Exponential Smoothing", "sec:es", ["Weighted average with decay", "Alpha near 1 -> responsive", "Alpha near 0 -> smooth"], "\\[ s_t = \\alpha x_t + (1-\\alpha)s_{t-1} \\]"),
        ],
        "exercises": [
            ("Window effect", "Increase window from 3 to 15: what happens?", ["Smoother, more lag."]),
            ("Alpha", "If alpha=0.9, smoothing is strong or weak?", ["Weak smoothing (very responsive)."]),
            ("Too much smoothing", "Why can too much smoothing harm forecasting?", ["It can hide real changes and add lag."]),
        ],
        "exit": "What is one sign that your smoothing window is too large?",
    },
    "ar_ma": {
        "outcomes": [
            "Explain AR(p) model idea",
            "Explain MA(q) model idea",
            "Differentiate AR vs MA intuition",
            "Define white noise (basic)",
        ],
        "sections": [
            ("AR", "sec:ar", ["Current value depends on past values", "AR(1): x_t = c + phi x_{t-1} + e_t", "Phi controls persistence"], "\\[ x_t = c + \\phi x_{t-1} + \\epsilon_t \\]"),
            ("MA", "sec:ma", ["Current value depends on past shocks", "MA(1): x_t = mu + e_t + theta e_{t-1}", "Captures short-term shock effects"], "\\[ x_t = \\mu + \\epsilon_t + \\theta\\epsilon_{t-1} \\]"),
        ],
        "exercises": [
            ("AR intuition", "If phi=0.8 and last value is high (ignore noise), what happens next?", ["Next value tends to be high too."]),
            ("MA intuition", "In MA(1), what drives the series: past values or past shocks?", ["Past shocks (errors)."]),
            ("White noise", "What is white noise?", ["Uncorrelated errors with mean 0 and constant variance."]),
        ],
        "exit": "How are AR and MA models different in what they remember?",
    },
    "arima": {
        "outcomes": [
            "Define ARIMA(p,d,q) at a high level",
            "Explain differencing (d) to remove trend",
            "Explain p and q meaning (AR and MA orders)",
            "Describe time-based train/test split for forecasting",
        ],
        "sections": [
            ("ARIMA", "sec:arima", ["p: AR order", "d: differencing order", "q: MA order"], None),
            ("Differencing", "sec:diff", ["First difference: y_t - y_{t-1}", "Often stabilizes mean", "Over-differencing adds noise"], None),
        ],
        "exercises": [
            ("Meaning of d", "What does d=1 mean?", ["First differencing once."]),
            ("Chronological split", "Why not random split in time series?", ["Random split leaks future information."]),
            ("Trend fix", "Series has strong upward trend. Name one simple step.", ["First differencing."]),
        ],
        "exit": "Why do we check residuals after fitting an ARIMA model?",
    },
    "stationarity": {
        "outcomes": [
            "Define stationarity (intuition)",
            "Recognize non-stationary patterns (trend/seasonality)",
            "Explain why stationarity matters for ARIMA-type models",
            "List basic fixes (differencing, transforms)",
        ],
        "sections": [
            ("Stationarity", "sec:stat", ["Mean/variance roughly constant", "Autocorrelation depends on lag only", "Trend/seasonality often implies non-stationarity"], None),
            ("Fixes", "sec:fix", ["Differencing removes trend", "Seasonal differencing removes seasonality", "Log transform can stabilize variance"], None),
        ],
        "exercises": [
            ("Trend", "Is a strong upward trend likely stationary?", ["No; mean changes over time."]),
            ("Variance change", "If fluctuations grow over time, is variance constant?", ["No; non-stationary variance."]),
            ("Fix choice", "Name one fix for non-stationary mean.", ["Differencing."]),
        ],
        "exit": "Why does non-stationarity make forecasting harder?",
    },
    "adf": {
        "outcomes": [
            "State null and alternative of ADF test (unit root)",
            "Interpret ADF p-value for stationarity decision",
            "Apply ADF to original and differenced series (idea)",
            "Explain why tests are not the only evidence (plots matter)",
        ],
        "sections": [
            ("ADF Test", "sec:adf", ["H0: unit root (non-stationary)", "H1: stationary", "Small p-value -> reject H0"], None),
            ("Interpretation", "sec:interp", ["If non-stationary, difference and test again", "Seasonality can require seasonal differencing", "Use ACF/PACF + diagnostics too"], None),
        ],
        "exercises": [
            ("ADF null", "What is H0 in ADF?", ["Unit root; non-stationary."]),
            ("Decision", "If p=0.02 at alpha=0.05, what do you conclude?", ["Reject H0; evidence of stationarity."]),
            ("Next step", "If p=0.6, what next step?", ["Difference and test again; consider seasonal differencing."]),
        ],
        "exit": "Why should we not rely on only one test to decide stationarity?",
    },
    "acf_pacf": {
        "outcomes": [
            "Define ACF and PACF (intuition)",
            "Use ACF/PACF patterns to guess AR and MA orders (rough)",
            "Recognize slow ACF decay as non-stationarity hint",
            "Explain why validation/diagnostics are still needed",
        ],
        "sections": [
            ("ACF", "sec:acf", ["Corr(x_t, x_{t-k}) vs lag k", "Slow decay can suggest non-stationarity", "ACF cutoff after q lags can suggest MA(q) (rough)"], None),
            ("PACF", "sec:pacf", ["Partial correlation after removing lower lags", "PACF cutoff after p lags can suggest AR(p) (rough)", "Use with caution and validate"], None),
        ],
        "exercises": [
            ("ACF pattern", "If ACF stays significant for many lags, what might it suggest?", ["Non-stationary; try differencing."]),
            ("PACF AR hint", "PACF cuts off after lag 2: what AR order to try?", ["Try AR(2) as starting point."]),
            ("Rules not perfect", "Why ACF/PACF rules are not perfect?", ["Finite sample noise, seasonality, mixed ARMA."]),
        ],
        "exit": "How do ACF and PACF help choose ARIMA orders p and q?",
    },
    "sarima": {
        "outcomes": [
            "Explain why diagnostics are needed after fitting",
            "Recognize residual autocorrelation as model issue",
            "Explain SARIMA seasonal terms at a high level",
            "Choose seasonal period s (weekly/monthly/yearly)",
        ],
        "sections": [
            ("Diagnostics", "sec:diag", ["Residuals should look like white noise", "Check residual ACF", "Check stability of variance"], None),
            ("SARIMA", "sec:sarima", ["ARIMA(p,d,q) x (P,D,Q,s)", "Seasonal differencing D", "s is the seasonal period"], None),
        ],
        "exercises": [
            ("Residual goal", "After fitting, what should residuals look like ideally?", ["White noise: no pattern, no autocorrelation."]),
            ("Seasonal period", "Daily data with weekly seasonality: what is s?", ["s=7"]),
            ("Why seasonal", "Why add seasonal terms?", ["To capture repeating seasonal dependence."]),
        ],
        "exit": "What is one residual symptom that suggests your model is inadequate?",
    },
}


LECTURES: list[Lecture] = [
    Lecture(
        unit=3,
        lecture=1,
        subtitle="Population, Sample, Sampling, Estimation and Confidence Intervals",
        topic="Population vs sample; sampling techniques; estimation; confidence intervals (overview).",
        demo_type="sampling_ci",
        outcomes=TYPE_LIBRARY["sampling_ci"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["sampling_ci"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["sampling_ci"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["sampling_ci"]["exit"],  # type: ignore[index]
    )
    ,
    Lecture(
        unit=3,
        lecture=2,
        subtitle="Hypothesis Testing (t-test): Concepts and Setup",
        topic="Null vs alternative; one-sample and two-sample t-tests; p-values and interpretation.",
        demo_type="ttest",
        outcomes=TYPE_LIBRARY["ttest"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["ttest"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["ttest"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["ttest"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=3,
        lecture=3,
        subtitle="Hypothesis Testing (t-test): Paired Test and Effect Size",
        topic="Paired t-test, mean difference, effect size, and interpretation.",
        demo_type="paired_ttest",
        outcomes=TYPE_LIBRARY["paired_ttest"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["paired_ttest"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["paired_ttest"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["paired_ttest"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=3,
        lecture=4,
        subtitle="Chi-square Tests (Goodness-of-Fit and Independence)",
        topic="Chi-square tests for counts: GOF and independence; expected counts; assumptions.",
        demo_type="chi_square",
        outcomes=TYPE_LIBRARY["chi_square"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["chi_square"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["chi_square"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["chi_square"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=3,
        lecture=5,
        subtitle="ANOVA (One-Way) and Post-hoc Intuition",
        topic="One-way ANOVA; F statistic; assumptions; post-hoc comparisons.",
        demo_type="anova",
        outcomes=TYPE_LIBRARY["anova"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["anova"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["anova"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["anova"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=3,
        lecture=6,
        subtitle="Non-parametric Tests and p-value Interpretation",
        topic="Rank-based tests and p-value interpretation; statistical vs practical significance.",
        demo_type="nonparametric",
        outcomes=TYPE_LIBRARY["nonparametric"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["nonparametric"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["nonparametric"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["nonparametric"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=3,
        lecture=7,
        subtitle="Case Exercise: Interpreting Hypothesis Testing Results",
        topic="Interpret published hypothesis testing results; emphasize CI, effect size, and pitfalls.",
        demo_type="case_interpretation",
        outcomes=TYPE_LIBRARY["case_interpretation"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["case_interpretation"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["case_interpretation"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["case_interpretation"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=1,
        subtitle="Correlation and Regression: Concepts",
        topic="Correlation vs regression concepts; causation warning; residual idea.",
        demo_type="corr_reg",
        outcomes=TYPE_LIBRARY["corr_reg"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["corr_reg"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["corr_reg"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["corr_reg"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=2,
        subtitle="Simple Linear Regression (OLS)",
        topic="Simple linear regression model, interpretation, residuals and R-squared.",
        demo_type="slr",
        outcomes=TYPE_LIBRARY["slr"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["slr"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["slr"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["slr"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=3,
        subtitle="Multiple Linear Regression",
        topic="Multiple predictors; partial effects; dummy variables; adjusted R-squared (overview).",
        demo_type="mlr",
        outcomes=TYPE_LIBRARY["mlr"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["mlr"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["mlr"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["mlr"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=4,
        subtitle="Polynomial Regression and Logistic Regression",
        topic="Polynomial regression for curvature; logistic regression for classification; basic evaluation metrics.",
        demo_type="poly_logit",
        outcomes=TYPE_LIBRARY["poly_logit"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["poly_logit"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["poly_logit"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["poly_logit"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=5,
        subtitle="Multicollinearity",
        topic="Multicollinearity: definition, symptoms, detection, and fixes.",
        demo_type="multicollinearity",
        outcomes=TYPE_LIBRARY["multicollinearity"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["multicollinearity"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["multicollinearity"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["multicollinearity"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=6,
        subtitle="VIF, AIC/BIC, Ridge and Lasso (Part 1)",
        topic="VIF concept; model selection criteria; ridge and lasso regularization (intro).",
        demo_type="vif_regularization",
        outcomes=TYPE_LIBRARY["vif_regularization"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["vif_regularization"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["vif_regularization"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["vif_regularization"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=7,
        subtitle="VIF, AIC/BIC, Ridge and Lasso (Part 2)",
        topic="Regularization continuation: bias-variance intuition, scaling, and choosing lambda (conceptually).",
        demo_type="vif_regularization",
        outcomes=TYPE_LIBRARY["vif_regularization"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["vif_regularization"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["vif_regularization"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["vif_regularization"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=8,
        subtitle="Cross-validation and Hyper-parameter Tuning",
        topic="Train/test split, k-fold cross-validation, and hyper-parameter tuning (grid/random search).",
        demo_type="cv_tuning",
        outcomes=TYPE_LIBRARY["cv_tuning"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["cv_tuning"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["cv_tuning"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["cv_tuning"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=4,
        lecture=9,
        subtitle="Case Study: End-to-End Regression Workflow",
        topic="End-to-end workflow: data -> model -> evaluation -> communication (case style).",
        demo_type="regression_case",
        outcomes=TYPE_LIBRARY["regression_case"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["regression_case"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["regression_case"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["regression_case"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=5,
        lecture=1,
        subtitle="Feature Selection, Engineering and Dimensionality Reduction (Intro)",
        topic="Intro to feature selection, feature engineering, and dimensionality reduction.",
        demo_type="feature_intro",
        outcomes=TYPE_LIBRARY["feature_intro"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["feature_intro"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["feature_intro"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["feature_intro"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=5,
        lecture=2,
        subtitle="Feature Selection Methods (Filter/Wrapper/Embedded)",
        topic="Filter, wrapper, and embedded feature selection methods (overview).",
        demo_type="feature_selection",
        outcomes=TYPE_LIBRARY["feature_selection"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["feature_selection"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["feature_selection"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["feature_selection"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=5,
        lecture=3,
        subtitle="Principal Component Analysis (PCA)",
        topic="PCA: variance-maximizing projection; explained variance; scaling.",
        demo_type="pca",
        outcomes=TYPE_LIBRARY["pca"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["pca"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["pca"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["pca"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=5,
        lecture=4,
        subtitle="Factor Analysis and Discriminant Analysis (LDA)",
        topic="Factor analysis (latent factors) and LDA (supervised separation).",
        demo_type="factor_lda",
        outcomes=TYPE_LIBRARY["factor_lda"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["factor_lda"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["factor_lda"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["factor_lda"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=5,
        lecture=5,
        subtitle="Kernel PCA and t-SNE",
        topic="Nonlinear dimensionality reduction: kernel PCA and t-SNE (visualization).",
        demo_type="kpca_tsne",
        outcomes=TYPE_LIBRARY["kpca_tsne"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["kpca_tsne"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["kpca_tsne"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["kpca_tsne"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=5,
        lecture=6,
        subtitle="Advanced Feature Engineering for Multivariate Data",
        topic="Interactions, aggregations, time features, and leakage avoidance.",
        demo_type="adv_feature_eng",
        outcomes=TYPE_LIBRARY["adv_feature_eng"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["adv_feature_eng"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["adv_feature_eng"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["adv_feature_eng"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=5,
        lecture=7,
        subtitle="Case Study: PCA + Clustering",
        topic="Case study: PCA + KMeans clustering; visualize and interpret clusters.",
        demo_type="pca_clustering",
        outcomes=TYPE_LIBRARY["pca_clustering"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["pca_clustering"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["pca_clustering"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["pca_clustering"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=1,
        subtitle="Time-series Concepts (Trend, Seasonality, Autocorrelation)",
        topic="Time series basics: components and autocorrelation.",
        demo_type="ts_intro",
        outcomes=TYPE_LIBRARY["ts_intro"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["ts_intro"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["ts_intro"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["ts_intro"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=2,
        subtitle="Smoothing (Moving Average and Exponential Smoothing)",
        topic="Smoothing techniques for time series (moving average, exponential smoothing).",
        demo_type="smoothing",
        outcomes=TYPE_LIBRARY["smoothing"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["smoothing"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["smoothing"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["smoothing"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=3,
        subtitle="AR and MA Models",
        topic="Autoregressive (AR) and moving average (MA) models (intro).",
        demo_type="ar_ma",
        outcomes=TYPE_LIBRARY["ar_ma"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["ar_ma"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["ar_ma"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["ar_ma"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=4,
        subtitle="Forecasting Fundamentals and ARIMA",
        topic="ARIMA models and forecasting workflow (overview).",
        demo_type="arima",
        outcomes=TYPE_LIBRARY["arima"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["arima"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["arima"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["arima"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=5,
        subtitle="Stationarity and Non-stationarity",
        topic="Stationarity concept; why it matters; fixes like differencing.",
        demo_type="stationarity",
        outcomes=TYPE_LIBRARY["stationarity"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["stationarity"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["stationarity"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["stationarity"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=6,
        subtitle="ADF Test for Stationarity",
        topic="ADF test (unit root) and interpretation.",
        demo_type="adf",
        outcomes=TYPE_LIBRARY["adf"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["adf"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["adf"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["adf"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=7,
        subtitle="ACF and PACF Interpretation",
        topic="ACF/PACF interpretation to guide AR/MA orders (rough).",
        demo_type="acf_pacf",
        outcomes=TYPE_LIBRARY["acf_pacf"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["acf_pacf"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["acf_pacf"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["acf_pacf"]["exit"],  # type: ignore[index]
    ),
    Lecture(
        unit=6,
        lecture=8,
        subtitle="Diagnostics and SARIMA Models",
        topic="Time-series diagnostics and SARIMA seasonal modeling (overview).",
        demo_type="sarima",
        outcomes=TYPE_LIBRARY["sarima"]["outcomes"],  # type: ignore[index]
        sections=TYPE_LIBRARY["sarima"]["sections"],  # type: ignore[index]
        exercises=TYPE_LIBRARY["sarima"]["exercises"],  # type: ignore[index]
        exit_question=TYPE_LIBRARY["sarima"]["exit"],  # type: ignore[index]
    ),
]


def _demo_wrapper(demo_type: str) -> str:
    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import sys
        from pathlib import Path

        lecture_dir = Path(__file__).resolve().parents[1]
        subject_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(subject_root / "_shared"))

        from demo_runner import run

        run("{demo_type}", lecture_dir)
        """
    ).strip()


def _make_slides(lec: Lecture) -> str:
    quick_links = [("Overview", "sec:overview")]
    quick_links.extend([(name, label) for (name, label, _bullets, _eq) in lec.sections])
    quick_links.extend([("Exercises", "sec:exercises"), ("Demo", "sec:demo"), ("Summary", "sec:summary")])

    parts: list[str] = [_slides_preamble(lec.unit, lec.lecture, lec.subtitle, quick_links)]
    parts.append("\\section{Overview}\n\\label{sec:overview}\n")
    parts.append(_frame_bullets("Learning Outcomes", lec.outcomes))

    for name, label, bullets, eq in lec.sections:
        parts.append(f"\\section{{{name}}}\n\\label{{{label}}}\n")
        parts.append(_frame_bullets(f"{name}: Key Points", bullets))
        if eq:
            parts.append(_frame_math(f"{name}: Key Formula", eq))

    parts.append("\\section{Exercises}\n\\label{sec:exercises}\n")
    for i, (title, prompt, sol) in enumerate(lec.exercises, start=1):
        parts.append(_frame_ex(f"Exercise {i}: {title}", prompt))
        parts.append(_frame_sol(f"Solution {i}", sol))

    parts.append("\\section{Demo}\n\\label{sec:demo}\n")
    parts.append(
        textwrap.dedent(
            """
            \\begin{frame}{Mini Demo (Python)}
              Run from the lecture folder:
              \\begin{center}
                \\texttt{python demo/demo.py}
              \\end{center}
              \\vspace{0.4em}
              Outputs:
              \\begin{itemize}
                \\item \\texttt{images/demo.png}
                \\item \\texttt{data/results.txt}
              \\end{itemize}
            \\end{frame}

            \\begin{frame}{Demo Output (Example)}
              \\begin{center}
              \\IfFileExists{../images/demo.png}{
                \\includegraphics[width=0.92\\linewidth]{demo.png}
              }{
                \\small (Run demo to generate: \\texttt{demo.png})
              }
              \\end{center}
            \\end{frame}
            """
        ).strip()
    )

    parts.append("\\section{Summary}\n\\label{sec:summary}\n")
    parts.append(_frame_bullets("Summary", ["Key definitions and the main formula.", "How to interpret results in context.", "How the demo connects to the theory."]))
    parts.append(_frame_ex("Exit Question", lec.exit_question))
    parts.append(_slides_end())
    return "\n\n".join(parts)


def main() -> None:
    for lec in LECTURES:
        lec_dir = lec.dir
        _write(lec_dir / "demo" / "demo.py", _demo_wrapper(lec.demo_type))
        _write(lec_dir / "latex" / "slides.tex", _make_slides(lec))

        note_sections = [(name, bullets) for (name, _label, bullets, _eq) in lec.sections]
        _write(
            lec_dir / "latex" / "notes.tex",
            _notes_tex(lec.unit, lec.lecture, lec.topic, lec.outcomes, note_sections, lec.exercises, lec.exit_question),
        )

    print(f"Generated {len(LECTURES)} lecture(s).")


if __name__ == "__main__":
    main()
