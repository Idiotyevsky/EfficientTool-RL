#!/usr/bin/env python3
"""Plot Nature-style curves from SearchAgent-RL verl artifacts.

Reads numeric rollouts/STEP.jsonl and validation/STEP.jsonl dumps, plus
optional verl console logs containing "step:N - metric:value". Missing
checkpoints are never interpolated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from efficienttool_rl.evaluation.verl_analysis import analyze_verl_behavior

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")
PAIR = re.compile(
    r"(?:^|\s)-\s+([^:\s]+):"
    r"([-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?)"
)
FIELDS = (
    "total_reward",
    "answer_reward",
    "task_reward",
    "marginal_evidence_reward",
    "current_beta",
    "wasted_search_penalty",
    "search_count",
    "useful_search_count",
    "wasted_search_count",
    "unique_support_count",
    "multi_search",
    "evidence_gain_per_search",
    "repeated_search_count",
    "em",
    "f1",
    "valid_answer",
)
ALIASES = {
    "total_reward": ("score", "reward"),
    "search_count": ("executed_search_calls",),
    "useful_search_count": ("useful_search_calls",),
    "wasted_search_count": ("wasted_search_calls",),
}
COLORS = {
    "blue": "#3B6FB6",
    "orange": "#D55E00",
    "green": "#198754",
    "purple": "#7B5AA6",
    "cyan": "#159EAA",
    "gray": "#687078",
    "red": "#B33A3A",
}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def parse_console_metrics(paths: Sequence[Path]) -> dict[int, dict[str, float]]:
    """Parse console metrics; later files win for duplicated resumed steps."""
    result: dict[int, dict[str, float]] = defaultdict(dict)
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = ANSI.sub("", raw)
                match = STEP.search(line)
                if not match:
                    continue
                step = int(match.group(1))
                result[step].update(
                    {name: float(value) for name, value in PAIR.findall(line)}
                )
                result[step].setdefault("training/global_step", float(step))
    return dict(sorted(result.items()))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _value(row: Mapping[str, object], field: str) -> float | None:
    for key in (field, *ALIASES.get(field, ())):
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def aggregate_jsonl(path: Path) -> dict[str, float]:
    """Aggregate trajectory fields and derive reward/group diagnostics."""
    rows = _read_jsonl(path)
    output: dict[str, float] = {}
    for field in FIELDS:
        values = [value for row in rows if (value := _value(row, field)) is not None]
        if not values:
            continue
        output[field] = sum(values) / len(values)
        if field in {"total_reward", "answer_reward", "task_reward"}:
            mean = output[field]
            output[f"{field}_std"] = math.sqrt(
                sum((value - mean) ** 2 for value in values) / len(values)
            )

    # Early Vanilla GRPO dumps predate the flattened search-counter fields,
    # but preserve the complete model/tool transcript. Reuse the canonical
    # analyzer to recover only metrics that are exactly observable without
    # supporting-title metadata. Explicit newer fields remain authoritative.
    if rows and all(
        isinstance(row.get("output"), str) and isinstance(row.get("gts"), str)
        for row in rows
    ):
        behavior = analyze_verl_behavior(rows)
        output.setdefault("search_count", behavior["avg_executed_search_calls"])
        output.setdefault("multi_search", behavior["multi_search_rate"])

    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prompt, score = row.get("input"), _value(row, "total_reward")
        if isinstance(prompt, str) and score is not None:
            groups[prompt].append(score)
    eligible = [scores for scores in groups.values() if len(scores) > 1]
    if eligible:
        output["zero_variance_group_ratio"] = sum(
            max(scores) - min(scores) <= 1e-12 for scores in eligible
        ) / len(eligible)
    output["trajectory_count"] = float(len(rows))
    output["group_count"] = float(len(eligible))
    return output


def load_run_directory(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    result: dict[str, dict[int, dict[str, float]]] = {"train": {}, "validation": {}}
    for split, name in (("train", "rollouts"), ("validation", "validation")):
        directory = path / name
        if not directory.is_dir():
            continue
        files = (item for item in directory.glob("*.jsonl") if item.stem.isdigit())
        for item in sorted(files, key=lambda value: int(value.stem)):
            result[split][int(item.stem)] = aggregate_jsonl(item)
    return result


def merge_run(
    directory: Path | None, logs: Sequence[Path]
) -> dict[str, dict[int, dict[str, float]]]:
    data = (
        load_run_directory(directory) if directory else {"train": {}, "validation": {}}
    )
    for step, metrics in parse_console_metrics(logs).items():
        data["train"].setdefault(step, {}).update(metrics)
    return data


def to_long_rows(
    runs: Mapping[str, dict[str, dict[int, dict[str, float]]]],
) -> list[dict[str, object]]:
    rows = []
    for run, splits in runs.items():
        for split, steps in splits.items():
            for step, metrics in steps.items():
                rows.extend(
                    {
                        "run": run,
                        "split": split,
                        "step": step,
                        "metric": metric,
                        "value": value,
                    }
                    for metric, value in metrics.items()
                    if math.isfinite(value)
                )
    return rows


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run", "split", "step", "metric", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("both LABEL and PATH are required")
    return label.strip(), Path(path).expanduser()


def _style() -> None:
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _series(frame, run: str, split: str, metric: str):
    return frame[
        (frame["run"] == run) & (frame["split"] == split) & (frame["metric"] == metric)
    ].sort_values("step")


def _plot(
    ax,
    frame,
    runs: Sequence[str],
    split: str,
    metric: str,
    label: str,
    color: str,
    smooth: int,
    percent: bool = False,
) -> bool:
    import seaborn as sns

    plotted = False
    for index, run in enumerate(runs):
        values = _series(frame, run, split, metric).copy()
        if values.empty:
            continue
        values["display"] = values["value"] * (100 if percent else 1)
        line_label = label if len(runs) == 1 else f"{run} · {label}"
        line_style = ("-", "--", ":", "-.")[index % 4]
        if split == "train" and smooth > 1 and len(values) > 2:
            ax.plot(
                values["step"],
                values["display"],
                color=color,
                alpha=0.20,
                linewidth=0.7,
                linestyle=line_style,
            )
            values["display"] = (
                values["display"].rolling(smooth, center=True, min_periods=1).mean()
            )
        sns.lineplot(
            data=values,
            x="step",
            y="display",
            ax=ax,
            color=color,
            label=line_label,
            linestyle=line_style,
            marker="o" if split == "validation" else None,
            estimator=None,
            errorbar=None,
        )
        plotted = True
    return plotted


def _finish(ax, title: str, ylabel: str, percent: bool = False) -> None:
    import seaborn as sns

    ax.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
    ax.title.set_ha("left")
    ax.title.set_position((0, 1.0))
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.65)
    ax.tick_params(direction="out", length=3, width=0.6)
    if percent:
        ax.set_ylim(-2, 102)
    sns.despine(ax=ax)


def _save(fig, stem: Path, formats: Sequence[str]) -> list[Path]:
    paths = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white")
        if suffix == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
            )
        paths.append(path)
    return paths


def plot_overview(frame, output: Path, formats: Sequence[str], smooth: int):
    import matplotlib.pyplot as plt

    runs = list(dict.fromkeys(frame["run"]))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    _plot(
        axes[0, 0],
        frame,
        runs,
        "train",
        "answer_reward",
        "Answer",
        COLORS["blue"],
        smooth,
    )
    _plot(
        axes[0, 0],
        frame,
        runs,
        "train",
        "total_reward",
        "Total",
        COLORS["orange"],
        smooth,
    )
    _finish(axes[0, 0], "a  Training reward", "Reward")
    _plot(axes[0, 1], frame, runs, "validation", "em", "EM", COLORS["blue"], 1, True)
    _plot(axes[0, 1], frame, runs, "validation", "f1", "F1", COLORS["orange"], 1, True)
    _finish(axes[0, 1], "b  Strict validation quality", "Score (%)", True)
    _plot(
        axes[1, 0],
        frame,
        runs,
        "train",
        "search_count",
        "Executed",
        COLORS["gray"],
        smooth,
    )
    _plot(
        axes[1, 0],
        frame,
        runs,
        "train",
        "useful_search_count",
        "Support-hit",
        COLORS["green"],
        smooth,
    )
    _plot(
        axes[1, 0],
        frame,
        runs,
        "train",
        "wasted_search_count",
        "No-new-support",
        COLORS["red"],
        smooth,
    )
    _finish(axes[1, 0], "c  Search behavior", "Calls per trajectory")
    _plot(
        axes[1, 1],
        frame,
        runs,
        "train",
        "multi_search",
        "Multi-search",
        COLORS["purple"],
        smooth,
        True,
    )
    _plot(
        axes[1, 1],
        frame,
        runs,
        "train",
        "valid_answer",
        "Valid answer",
        COLORS["cyan"],
        smooth,
        True,
    )
    _finish(axes[1, 1], "d  Episode outcomes", "Rate (%)", True)
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
    paths = _save(fig, output / "training_overview", formats)
    plt.close(fig)
    return paths


def plot_panels(
    frame,
    output: Path,
    formats: Sequence[str],
    smooth: int,
    stem: str,
    panels: Sequence[tuple[str, str, str]],
    shape: tuple[int, int],
):
    import matplotlib.pyplot as plt

    runs = list(dict.fromkeys(frame["run"]))
    if not any(not frame[frame["metric"] == metric].empty for metric, _, _ in panels):
        return []
    height = 4.6 if shape[0] == 2 else 2.35
    fig, axes = plt.subplots(
        *shape, figsize=(7.2, height), constrained_layout=True, squeeze=False
    )
    for index, (metric, title, color) in enumerate(panels):
        ax = axes.flat[index]
        if _plot(ax, frame, runs, "train", metric, title, color, smooth):
            _finish(ax, f"{chr(97 + index)}  {title}", title)
            ax.legend(loc="best")
        else:
            ax.set_visible(False)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    paths = _save(fig, output / stem, formats)
    plt.close(fig)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIR")
    parser.add_argument("--log", action="append", default=[], metavar="LABEL=FILE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    args = parser.parse_args(argv)
    if args.smooth < 1:
        parser.error("--smooth must be at least 1")
    run_paths = dict(_named_path(item) for item in args.run)
    logs: dict[str, list[Path]] = defaultdict(list)
    for item in args.log:
        label, path = _named_path(item)
        logs[label].append(path)
    labels = list(dict.fromkeys([*run_paths, *logs]))
    if not labels:
        parser.error("provide at least one --run LABEL=DIR or --log LABEL=FILE")
    for label, path in run_paths.items():
        if not path.is_dir():
            parser.error(f"missing run directory for {label}: {path}")
    for label, paths in logs.items():
        for path in paths:
            if not path.is_file():
                parser.error(f"missing log for {label}: {path}")
    runs = {
        label: merge_run(run_paths.get(label), logs.get(label, [])) for label in labels
    }
    rows = to_long_rows(runs)
    if not rows:
        parser.error("no numeric metrics found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "metrics_long.csv"
    write_csv(rows, csv_path)
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit(
            'install plotting dependencies: pip install -e ".[plot]"'
        ) from error
    _style()
    frame = pd.DataFrame(rows)
    written = [
        csv_path,
        *plot_overview(frame, args.output_dir, args.formats, args.smooth),
    ]
    written += plot_panels(
        frame,
        args.output_dir,
        args.formats,
        args.smooth,
        "optimization_health",
        (
            ("actor/grad_norm", "Gradient norm", COLORS["blue"]),
            ("actor/entropy", "Policy entropy", COLORS["green"]),
            ("actor/kl_loss", "KL loss", COLORS["orange"]),
            ("critic/advantages/abs_mean", "Mean |advantage|", COLORS["purple"]),
            (
                "zero_variance_group_ratio",
                "Zero-variance groups",
                COLORS["red"],
            ),
            ("timing_s/step", "Step time (s)", COLORS["gray"]),
        ),
        (2, 3),
    )
    written += plot_panels(
        frame,
        args.output_dir,
        args.formats,
        args.smooth,
        "reward_components",
        (
            ("current_beta", "Evidence coefficient", COLORS["purple"]),
            ("marginal_evidence_reward", "Marginal evidence", COLORS["green"]),
            ("wasted_search_penalty", "Search penalty", COLORS["red"]),
        ),
        (1, 3),
    )
    manifest = {
        "runs": {
            label: {
                "artifact": (run_paths[label].name if label in run_paths else None),
                "console_logs": [path.name for path in logs.get(label, [])],
                "train_steps": len(runs[label]["train"]),
                "validation_steps": len(runs[label]["validation"]),
            }
            for label in labels
        },
        "smoothing_window": args.smooth,
        "outputs": [path.name for path in written],
        "note": (
            "Validation points are observed; missing checkpoints are not interpolated."
        ),
    }
    manifest_path = args.output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(written)} data/figure files and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
