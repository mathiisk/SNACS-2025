from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


SAMPLER_ABBREV = {
    "random_node": "RN",
    "random_degree": "RDN",
    "random_pagerank": "RPR",
    "random_edge": "RE",
    "random_node_edge": "RNE",
    "random_hybrid": "HYB",
    "random_walk": "RW",
    "random_jump": "RJ",
    "forest_fire": "FF",
    "random_neigbour": "RNN",
}


def _aggregate_mean_and_std(group: pd.DataFrame, mean_col: str, std_col: str) -> pd.Series:
    means = group[mean_col].to_numpy(dtype=float)
    stds = group[std_col].to_numpy(dtype=float)

    mean = float(np.mean(means)) if means.size else np.nan
    between_var = float(np.var(means, ddof=0)) if means.size else 0.0
    within_var = float(np.mean(stds**2)) if stds.size else 0.0
    std = float(np.sqrt(between_var + within_var))
    return pd.Series({"mean": mean, "std": std})


def _project_root_from_this_file(this_file: str | Path) -> Path:
    return Path(this_file).resolve().parents[1]


def load_results_csvs(
    root_dir: str | Path,
    pattern: str = "results_*_rwfixed_fullgraph.csv",
) -> pd.DataFrame:
    root_dir = Path(root_dir)
    csv_paths = sorted(root_dir.rglob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' under: {root_dir.resolve()}")

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["source_path"] = str(p)
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)

    required = {"dataset", "model", "sampler", "ratio", "test_f1_mean", "test_f1_std"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    results = results.dropna(subset=["dataset", "model", "sampler", "ratio", "test_f1_mean"])
    results["ratio"] = pd.to_numeric(results["ratio"], errors="coerce")
    results["test_f1_mean"] = pd.to_numeric(results["test_f1_mean"], errors="coerce")
    results["test_f1_std"] = pd.to_numeric(results["test_f1_std"], errors="coerce").fillna(0.0)
    results = results.dropna(subset=["ratio", "test_f1_mean"])
    return results


def plot_sampler_retention(
    results: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
    ratios: list[float] | None = None,
    score_mean_col: str = "test_f1_mean",
    score_std_col: str = "test_f1_std",
    full_graph_sampler: str = "full_graph",
    include_full_graph_curve: bool = False,
    anchor_ratio1: bool = True,
    plot_loss: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    model_rows = results[results["model"].astype(str) == str(model_name)].copy()
    if model_rows.empty:
        raise ValueError(f"No rows for model='{model_name}'")

    model_rows["dataset"] = model_rows["dataset"].astype(str)
    model_rows["sampler"] = model_rows["sampler"].astype(str)

    per = (
        model_rows.groupby(["dataset", "sampler", "ratio"], as_index=False)
        .apply(lambda g: _aggregate_mean_and_std(g, score_mean_col, score_std_col))
        .reset_index()
    )
    per = per.rename(columns={"mean": "score_mean", "std": "score_std"})[
        ["dataset", "sampler", "ratio", "score_mean", "score_std"]
    ]

    baseline = per[(np.isclose(per["ratio"], 1.0)) & (per["sampler"] == full_graph_sampler)].copy()
    if baseline.empty:
        raise ValueError(f"Need baseline rows: sampler='{full_graph_sampler}' at ratio==1 per dataset")

    baseline_by_ds = baseline.groupby("dataset", as_index=False).agg(baseline_score=("score_mean", "mean"))
    normalized = per.merge(baseline_by_ds, on="dataset", how="inner")
    normalized["retention"] = normalized["score_mean"].to_numpy(float) / normalized["baseline_score"].to_numpy(float)

    if ratios is not None:
        ratio_levels = sorted(set(float(r) for r in ratios) | {1.0})
        normalized = normalized[normalized["ratio"].apply(lambda r: any(np.isclose(r, rr) for rr in ratio_levels))].copy()
    else:
        ratio_levels = sorted(set(normalized["ratio"].tolist()))
        if 1.0 not in ratio_levels:
            ratio_levels.append(1.0)

    avg = normalized.groupby(["sampler", "ratio"], as_index=False).agg(mean_retention=("retention", "mean"))

    if not include_full_graph_curve:
        avg = avg[avg["sampler"] != full_graph_sampler].copy()

    if anchor_ratio1:
        samplers = sorted(avg["sampler"].unique().tolist())
        anchor = pd.DataFrame({"sampler": samplers, "ratio": 1.0, "mean_retention": 1.0})
        avg = pd.concat([avg, anchor], ignore_index=True).drop_duplicates(subset=["sampler", "ratio"], keep="first")

    x_positions = {r: i for i, r in enumerate(ratio_levels)}

    samplers = sorted(avg["sampler"].unique().tolist())
    if sns is not None:
        palette = sns.color_palette("tab10", n_colors=max(3, len(samplers)))
        sampler_to_color = {s: palette[i % len(palette)] for i, s in enumerate(samplers)}
    else:
        cmap = plt.get_cmap("tab10")
        sampler_to_color = {s: cmap(i % 10) for i, s in enumerate(samplers)}

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)

    ref_y = 0.0 if plot_loss else 1.0
    ax.axhline(ref_y, linestyle="-", linewidth=1.2, alpha=0.6, color="black")

    for s, frame in avg.groupby("sampler"):
        frame = frame.sort_values("ratio")
        rvals = frame["ratio"].to_numpy(float)
        x = np.array([x_positions.get(r, np.nan) for r in rvals], dtype=float)
        y = frame["mean_retention"].to_numpy(float)

        valid = ~np.isnan(x)
        x, y = x[valid], y[valid]
        if plot_loss:
            y = 1.0 - y

        ax.plot(x, y, marker="o", markersize=5, linewidth=2.2, alpha=0.95, color=sampler_to_color[s])

    ax.set_xticks(range(len(ratio_levels)))
    ax.set_xticklabels([f"{int(round(r * 100))}%" for r in ratio_levels])
    ax.set_xlabel("Sampling ratio")
    ax.set_ylabel("1 - Test F1 retention (vs. full graph)" if plot_loss else "Test F1 retention (vs. full graph)")

    legend_labels = {s: SAMPLER_ABBREV.get(s, s) for s in samplers}
    handles = [
        Patch(facecolor=sampler_to_color[s], edgecolor=sampler_to_color[s], label=legend_labels[s])
        for s in samplers
    ]
    handles.append(Line2D([0], [0], color="black", linestyle="-", linewidth=1.2, alpha=0.6, label="y=1" if not plot_loss else "y=0"))

    ax.legend(
        handles=handles,
        ncol=3,
        fontsize=15,
        frameon=True,
        framealpha=0.95,
        loc="best",
        borderpad=1.1,
        labelspacing=0.8,
        handlelength=1.6,
        handletextpad=0.8,
        columnspacing=1.2,
    )

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run(
    model: str,
    results_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    ratios: list[float] | None = None,
    include_full_graph_curve: bool = False,
    anchor_ratio1: bool = True,
    plot_loss: bool = False,
) -> Path:
    project_root = _project_root_from_this_file(__file__)
    results_dir = Path(results_dir) if results_dir is not None else (project_root / "results")
    out_dir = Path(out_dir) if out_dir is not None else (project_root / "results" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_csvs(results_dir)

    out_path = out_dir / f"{model}_retention_vs_full.png"
    return plot_sampler_retention(
        results=results,
        model_name=model,
        output_path=out_path,
        ratios=ratios if ratios is not None else [0.05, 0.10, 0.30, 0.50, 1.0],
        include_full_graph_curve=include_full_graph_curve,
        anchor_ratio1=anchor_ratio1,
        plot_loss=plot_loss,
    )


if __name__ == "__main__":
    run(model="gcn")
    run(model="mlp")