"""Lab 1 — baselines + system measurements + deployment-fit table + report.md.

Usage (from repo root, with .venv active):  python lab01/src/run_lab01.py

Produces in lab01/results/:
  versions.txt, baseline_accuracy.csv, measurements.csv, deployment_fit.csv,
  model_logreg.joblib, model_rf.joblib, model.joblib (copy of the RF model),
  system_cost.png
and writes lab01/report.md
"""
import os
import random
import shutil
import statistics
import threading
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import print_versions  # noqa: E402  (same folder)

SEED = 42
LR_NOT_CONVERGED = False
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS = LAB_DIR / "results"
N_TRAIN_RUNS = 5
N_INFER_RUNS = 100

BUDGETS = {  # name: (max memory MB, max latency ms, max model size KB, memory text)
    "Cloud": (float("inf"), 100.0, 500 * 1024, ">= 1 GB"),
    "Edge": (1024.0, 50.0, 50 * 1024, "256-1024 MB"),
    "Mobile": (256.0, 20.0, 10 * 1024, "64-256 MB"),
    "TinyML": (0.25, 10.0, 100.0, "<= 256 KB"),
}


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    try:  # torch is in requirements; not needed for this lab but seeded per global rules
        import torch

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass


class PeakRSS:
    """Samples the process RSS in a background thread; reports peak in MB."""

    def __init__(self, interval: float = 0.001):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.start = self.proc.memory_info().rss
        self.peak = self.start
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.peak_mb = self.peak / 2**20
        self.delta_mb = (self.peak - self.start) / 2**20


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }


def measure(name, X_tr, X_te, y_tr, y_te):
    factory = lambda: make_models()[name]  # noqa: E731

    # Training time: 1 warm-up + 5 timed runs, median
    factory().fit(X_tr, y_tr)
    train_times = []
    for _ in range(N_TRAIN_RUNS):
        m = factory()
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        train_times.append(time.perf_counter() - t0)

    # Peak memory during training (fresh model)
    global LR_NOT_CONVERGED
    with PeakRSS() as mem_train, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = factory().fit(X_tr, y_tr)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        LR_NOT_CONVERGED = True

    acc = accuracy_score(y_te, model.predict(X_te))

    # Single-sample inference: warm-up + 100 timed runs, median
    sample = X_te[:1]
    model.predict(sample)
    infer_times = []
    with PeakRSS() as mem_infer:
        for _ in range(N_INFER_RUNS):
            t0 = time.perf_counter()
            model.predict(sample)
            infer_times.append(time.perf_counter() - t0)

    # Model size on disk
    short = "logreg" if name.startswith("Logistic") else "rf"
    path = RESULTS / f"model_{short}.joblib"
    joblib.dump(model, path)
    size_b = path.stat().st_size

    return model, {
        "model": name,
        "test_accuracy": round(acc, 4),
        "train_time_median_ms": statistics.median(train_times) * 1000,
        "train_time_min_ms": min(train_times) * 1000,
        "train_time_max_ms": max(train_times) * 1000,
        "infer_latency_median_ms": statistics.median(infer_times) * 1000,
        "infer_latency_p95_ms": float(np.percentile(infer_times, 95)) * 1000,
        "model_size_bytes": size_b,
        "model_size_kb": size_b / 1024,
        "peak_rss_train_mb": mem_train.peak_mb,
        "train_mem_increase_mb": mem_train.delta_mb,
        "peak_rss_infer_mb": mem_infer.peak_mb,
        "infer_mem_increase_mb": mem_infer.delta_mb,
    }


def deployment_fit(row):
    """Check each target. Memory need = peak process RSS during inference (whole Python process)."""
    out = []
    for target, (max_mem, max_lat, max_kb, mem_txt) in BUDGETS.items():
        mem_ok = row["peak_rss_infer_mb"] <= max_mem
        lat_ok = row["infer_latency_median_ms"] <= max_lat
        size_ok = row["model_size_kb"] <= max_kb
        fails = [n for n, ok in [("memory", mem_ok), ("latency", lat_ok), ("size", size_ok)] if not ok]
        out.append({
            "model": row["model"], "target": target,
            "memory_ok": mem_ok, "latency_ok": lat_ok, "size_ok": size_ok,
            "fits": not fails, "fails_on": ", ".join(fails) or "-",
        })
    return out


def plot(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("infer_latency_median_ms", "Single-sample latency, ms (median of 100)",
         [(20, "Mobile 20 ms"), (10, "TinyML 10 ms")], "log"),
        ("model_size_kb", "Model size on disk, KB",
         [(100, "TinyML 100 KB"), (10 * 1024, "Mobile 10 MB")], "log"),
        ("train_time_median_ms", "Training time, ms (median of 5)", [], "linear"),
    ]
    colors = ["#3b6ea5", "#d08c2b"]
    for ax, (col, title, lines, scale) in zip(axes, specs):
        ax.bar(df["model"], df[col], color=colors)
        for v, label in lines:
            ax.axhline(v, ls="--", lw=1, color="grey")
            ax.text(-0.45, v * 1.15, label, va="bottom", fontsize=8, color="grey")
        ax.set_yscale(scale)
        if lines:
            ax.set_ylim(top=max(v for v, _ in lines) * 4)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "system_cost.png", dpi=150)
    plt.close(fig)


def md_table(df, cols, headers, fmt):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(df, fit, cpu_info):
    lr = df.set_index("model").loc["LogisticRegression"]
    rf = df.set_index("model").loc["RandomForest"]
    fit_pivot = fit.pivot(index="target", columns="model", values="fits").loc[list(BUDGETS)]
    fails = fit.pivot(index="target", columns="model", values="fails_on").loc[list(BUDGETS)]

    def cell(t, m):
        return "yes" if fit_pivot.loc[t, m] else f"no ({fails.loc[t, m]})"

    fit_lines = ["| Target | Memory | Latency | Model size | LogReg | Random Forest |",
                 "|---|---|---|---|---|---|"]
    for t, (mm, ml, mk, mt) in BUDGETS.items():
        size_txt = f"{mk / 1024:g} MB" if mk >= 1024 else f"{mk:g} KB"
        fit_lines.append(f"| {t} | {mt} | <= {ml:g} ms | <= {size_txt} | "
                         f"{cell(t, 'LogisticRegression')} | {cell(t, 'RandomForest')} |")

    lat_ratio = rf["infer_latency_median_ms"] / lr["infer_latency_median_ms"]
    size_ratio = rf["model_size_bytes"] / lr["model_size_bytes"]
    train_ratio = rf["train_time_median_ms"] / lr["train_time_median_ms"]
    acc_diff = abs(rf["test_accuracy"] - lr["test_accuracy"]) * 100
    winner = "Random Forest" if rf["test_accuracy"] > lr["test_accuracy"] else "Logistic Regression"
    loser = "Logistic Regression" if winner == "Random Forest" else "Random Forest"
    rss = lr["peak_rss_infer_mb"]
    max_delta = max(lr["train_mem_increase_mb"], rf["train_mem_increase_mb"],
                    lr["infer_mem_increase_mb"], rf["infer_mem_increase_mb"])

    if train_ratio >= 1.5:
        train_sent = (f"Training the forest took about {train_ratio:.0f} times longer "
                      f"({rf['train_time_median_ms']:.0f} ms vs {lr['train_time_median_ms']:.0f} ms).")
    elif train_ratio <= 1 / 1.5:
        train_sent = (f"Surprisingly, the forest trained faster ({rf['train_time_median_ms']:.0f} ms vs "
                      f"{lr['train_time_median_ms']:.0f} ms). I think this is because LogReg runs all 1000 "
                      f"iterations on unscaled data.")
    else:
        train_sent = (f"Training time was actually close ({lr['train_time_median_ms']:.0f} ms for LogReg and "
                      f"{rf['train_time_median_ms']:.0f} ms for the forest). I expected LogReg to be much "
                      f"faster, but on unscaled data it runs all 1000 iterations and does not converge, "
                      f"so it loses its advantage here.")

    conv = ("With raw (unscaled) features LogReg hit the 1000 iteration limit and sklearn gave a "
            "ConvergenceWarning. I left the settings as in the task, but a StandardScaler would probably "
            "fix it and make training faster."
            if LR_NOT_CONVERGED else "")

    rf_tiny = ("Random Forest is out anyway, the file alone is "
               f"{rf['model_size_kb']:.0f} KB, more than the 100 KB limit."
               if rf["model_size_kb"] > 100 else
               "Random Forest is small enough by size, but it still needs Python to run.")

    if fit_pivot.loc["Mobile"].all():
        mobile_sent = f"by the numbers both fit (the process is ~{rss:.0f} MB, below 256 MB)."
    elif not fit_pivot.loc["Mobile"].any():
        mobile_sent = (f"on my machine neither fits, mostly because of memory: the Python process is "
                       f"~{rss:.0f} MB and the limit is 256 MB.")
    else:
        ok = "LogReg" if fit_pivot.loc["Mobile", "LogisticRegression"] else "Random Forest"
        mobile_sent = f"only {ok} fits on my machine."

    def tiers(m):
        return ", ".join(t for t in BUDGETS if fit_pivot.loc[t, m]) or "none"

    measure_tbl = md_table(
        df,
        ["model", "test_accuracy", "train_time_median_ms", "infer_latency_median_ms",
         "model_size_bytes", "model_size_kb", "peak_rss_train_mb", "peak_rss_infer_mb"],
        ["Model", "Accuracy", "Train time, ms", "Latency, ms", "Size, bytes", "Size, KB",
         "Peak RAM train, MB", "Peak RAM inference, MB"],
        {"test_accuracy": "{:.4f}", "train_time_median_ms": "{:.1f}",
         "infer_latency_median_ms": "{:.3f}", "model_size_bytes": "{:,}", "model_size_kb": "{:.2f}",
         "peak_rss_train_mb": "{:.1f}", "peak_rss_infer_mb": "{:.1f}"},
    )

    report = f"""# Lab 1: Environment and first system measurements

## Goal

The goal of this lab was to set up a clean Python environment and to see how much two simple models
actually "cost" to run, not only how accurate they are. I trained Logistic Regression and Random Forest
on the Breast Cancer Wisconsin dataset, measured time, memory and model size, and then checked where
each model could be deployed (Cloud, Edge, Mobile, TinyML).

## Method

I created a venv with Python 3.11.8 and installed the pinned versions from `requirements.txt`.
The script `src/print_versions.py` saves all versions to `results/versions.txt`.

One thing did not work out of the box: pip could not install the requirements because `mlflow==2.14.1`
needs `pyarrow` lower than 16, but the list asks for `pyarrow==16.1.0`. I changed only pyarrow to
15.0.2, everything else is exactly as required.

Setup:

- Data: `load_breast_cancer`, 569 samples, 30 features. Split 70/30, stratified, `random_state=42`
  (398 train, 171 test).
- Models: `LogisticRegression(max_iter=1000, random_state=42)` and
  `RandomForestClassifier(n_estimators=100, random_state=42)`.
- All seeds are 42.
- Training time: one warm-up run, then 5 runs, I report the median.
- Latency: prediction for one sample, one warm-up, then 100 runs, median.
- Model size: saved with `joblib.dump`, size of the file.
- Memory: I used psutil and checked the process RSS every 1 ms in a separate thread during training
  and during inference. This is memory of the whole Python process, not only the model.
- Machine: {cpu_info}.

{conv}

The code is in `src/run_lab01.py`, all outputs are in `results/`.

## Results

{measure_tbl}

![System cost](results/system_cost.png)

During training and inference the memory grew by less than {max_delta + 0.5:.0f} MB. Almost all of the
~{rss:.0f} MB is just Python with numpy, scipy and sklearn loaded.

Deployment check (a model fits only if memory, latency and size are all inside the budget):

{chr(10).join(fit_lines)}

Where can each model run:

- Cloud and Edge are no problem for both models. Even the forest predicts in
  {rf['infer_latency_median_ms']:.1f} ms, while the limit is 50 ms for Edge. On something like a
  Raspberry Pi it will be slower than on my laptop, but there is still a lot of room.
- Mobile: {mobile_sent} Random Forest is closer to
  the limits, its latency is about {lat_ratio:.0f}x higher and the file is about {size_ratio:.0f}x bigger.
  Also, nobody runs a full Python on a phone, so in real life the model would be exported to something
  like ONNX.
- TinyML: neither model fits as I measured it, because 256 KB of memory is nothing compared to the
  ~{rss:.0f} MB Python process. But LogReg itself is only 30 weights and one bias, so if I rewrote the
  prediction in C it would easily fit on a microcontroller. {rf_tiny}

So my decision: LogReg can go to {tiers('LogisticRegression')} (and TinyML after porting to C),
Random Forest to {tiers('RandomForest')}.

## Conclusions

1. The accuracy of both models is almost the same, {winner} is better only by {acc_diff:.2f}
   percentage points ({lr['test_accuracy']:.4f} vs {rf['test_accuracy']:.4f}). But the price is very
   different: the forest is about {lat_ratio:.0f}x slower on one prediction and about {size_ratio:.0f}x
   bigger. {train_sent} For a small table dataset like this I would choose LogReg.

2. What surprised me most is memory. The models themselves add almost nothing, but Python with the
   libraries already takes ~{rss:.0f} MB. So when we decide where to deploy, we need to look at the whole
   runtime, not only at the model file. This is the reason TinyML is impossible here, not the models.

3. Timing needs repeated runs. My training times jumped from {lr['train_time_min_ms']:.0f} to
   {lr['train_time_max_ms']:.0f} ms for LogReg and from {rf['train_time_min_ms']:.0f} to
   {rf['train_time_max_ms']:.0f} ms for the forest between runs. If I took just one run, I could
   easily get wrong conclusions, so warm-up and median really make sense.
"""
    (LAB_DIR / "report.md").write_text(report, encoding="utf-8")


def main():
    # Raw unscaled features + max_iter=1000 (as the manual prescribes) -> lbfgs hits the limit.
    # We keep the prescribed config, silence the console spam and document it in the report.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_seeds()
    print_versions.main()
    print()

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    rows = []
    for name in ["LogisticRegression", "RandomForest"]:
        print(f"Measuring {name} ...")
        model, row = measure(name, X_tr, X_te, y_tr, y_te)
        rows.append(row)
        if name == "RandomForest":
            shutil.copy(RESULTS / "model_rf.joblib", RESULTS / "model.joblib")

    df = pd.DataFrame(rows)
    df[["model", "test_accuracy"]].to_csv(RESULTS / "baseline_accuracy.csv", index=False,
                                          float_format="%.4f")
    df.round(4).to_csv(RESULTS / "measurements.csv", index=False)
    fit = pd.DataFrame([r for _, row in df.iterrows() for r in deployment_fit(row)])
    fit.to_csv(RESULTS / "deployment_fit.csv", index=False)
    plot(df)

    import platform
    cpu_info = f"{platform.processor() or platform.machine()}, {psutil.cpu_count(logical=True)} logical CPUs, " \
               f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM, {platform.system()}"
    write_report(df, fit, cpu_info)

    print(df[["model", "test_accuracy", "train_time_median_ms", "infer_latency_median_ms",
              "model_size_kb", "peak_rss_infer_mb"]].round(3).to_string(index=False))
    print("\n", fit[["model", "target", "fits", "fails_on"]].to_string(index=False))
    print(f"\nDone. Report written to {LAB_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
