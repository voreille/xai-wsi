"""Learning curves of tile-level runs from the local W&B files, without the W&B server.

W&B keeps each run's history in ``<wandb-dir>/run-<date>-<id>/run-<id>.wandb`` (a
LevelDB-style log of protobuf records). This reads the per-epoch ``history`` records of the
fit stage (the directory with validation epochs; the test stage resumes the same id in a
second directory) and groups runs by id without their ``-s<seed>`` suffix.

    uv run python -m xaiwsi.evaluation.wandb_curves \\
        tilemean=coh-1024-luad-percohort gated=coh-1105gated-luad-percohort \\
        t0=coh-1105nca-t0-luad-percohort t16=coh-1105nca-t16-luad-percohort \\
        --output outputs/nca_runs/curves

Writes ``<output>.csv`` (run, group, epoch, train/val loss, val median Spearman of the
monitored cohort), ``<output>.md`` (per group, mean over seeds: best val Spearman and its
epoch, epoch of the lowest val loss, val Spearman at a few epochs, epochs run) and
``<output>.png`` (curves, mean over seeds, seeds as thin lines).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import struct
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from xaiwsi.evaluation.cohort_probe import markdown_table

_BLOCK = 32768
_FULL, _FIRST, _MIDDLE, _LAST = 1, 2, 3, 4
# Default categorical palette, fixed order (one slot per group, never cycled).
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def wandb_records(path: Path) -> Iterator[bytes]:
    """Raw records of a ``.wandb`` file (7-byte header, 32 KiB blocks of chunked records)."""
    data = Path(path).read_bytes()
    if data[:4] != b":W&B":
        raise ValueError(f"{path}: not a W&B transaction log")
    pos, pending = 7, b""
    while pos + 7 <= len(data):
        left = _BLOCK - pos % _BLOCK
        if left < 7:  # block trailer
            pos += left
            continue
        _checksum, length, kind = struct.unpack("<IHB", data[pos : pos + 7])
        if kind == 0 and length == 0:  # zero padding up to the next block
            pos += left
            continue
        chunk = data[pos + 7 : pos + 7 + length]
        pos += 7 + length
        if kind == _FULL:
            yield chunk
        elif kind == _FIRST:
            pending = chunk
        elif kind == _MIDDLE:
            pending += chunk
        elif kind == _LAST:
            yield pending + chunk
        else:
            raise ValueError(f"{path}: unknown record type {kind} at byte {pos}")


def wandb_history(path: Path) -> list[dict]:
    """The logged history rows of one run file, in order."""
    from wandb.proto import wandb_internal_pb2 as pb

    rows = []
    for raw in wandb_records(path):
        record = pb.Record()
        record.ParseFromString(raw)
        if record.WhichOneof("record_type") == "history":
            rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in record.history.item})
    return rows


def epoch_table(rows: list[dict], monitor: str) -> pd.DataFrame:
    """One row per epoch: train_loss, val_loss, val_spearman (``monitor``)."""
    by_epoch: dict[int, dict] = {}
    for row in rows:
        if "epoch" not in row:
            continue
        entry = by_epoch.setdefault(int(row["epoch"]), {})
        if "train/loss" in row:
            entry["train_loss"] = row["train/loss"]
        if monitor in row:
            entry["val_spearman"] = row[monitor]
            entry["val_loss"] = row.get("val/loss")
    frame = pd.DataFrame.from_dict(by_epoch, orient="index").rename_axis("epoch").reset_index()
    return frame.dropna(subset=["val_spearman"]) if "val_spearman" in frame else frame.iloc[0:0]


def fit_history(wandb_dir: Path, run_id: str, monitor: str) -> pd.DataFrame:
    """Epoch table of the run's fit stage (the directory with the most validation epochs)."""
    best = None
    for path in sorted(wandb_dir.glob(f"run-*-{run_id}/run-{run_id}.wandb")):
        table = epoch_table(wandb_history(path), monitor)
        if best is None or len(table) > len(best):
            best = table
    if best is None or best.empty:
        raise SystemExit(f"no fit history for {run_id} in {wandb_dir}")
    return best


def summarize(curves: pd.DataFrame, at_epochs: list[int]) -> pd.DataFrame:
    rows = []
    for (group, run), frame in curves.groupby(["group", "run"], sort=False):
        frame = frame.sort_values("epoch")
        best = frame.loc[frame["val_spearman"].idxmax()]
        row = {
            "group": group,
            "run": run,
            "epochs_run": int(frame["epoch"].max()) + 1,
            "best_val_spearman": best["val_spearman"],
            "best_epoch": int(best["epoch"]),
            "min_val_loss_epoch": int(frame.loc[frame["val_loss"].idxmin(), "epoch"]),
            "last_train_loss": frame["train_loss"].iloc[-1],
        }
        for epoch in at_epochs:
            match = frame.loc[frame["epoch"] == epoch, "val_spearman"]
            row[f"val_spearman_e{epoch}"] = match.iloc[0] if len(match) else np.nan
        rows.append(row)
    per_run = pd.DataFrame(rows)
    numeric = per_run.columns.drop(["group", "run"])
    summary = per_run.groupby("group", sort=False)[list(numeric)].mean()
    summary.insert(0, "n_runs", per_run.groupby("group", sort=False).size())
    return summary.reset_index()


def plot(curves: pd.DataFrame, groups: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(groups) > len(_PALETTE):
        raise SystemExit(f"at most {len(_PALETTE)} groups per figure")
    panels = [("val_spearman", "Val median Spearman"), ("val_loss", "Val loss (MSE)"), ("train_loss", "Train loss")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, constrained_layout=True)
    for ax, (column, title) in zip(axes, panels, strict=True):
        for color, group in zip(_PALETTE, groups, strict=False):
            frame = curves[curves["group"] == group]
            for _, run in frame.groupby("run"):
                ax.plot(run["epoch"], run[column], color=color, linewidth=0.8, alpha=0.3)
            mean = frame.groupby("epoch")[column].mean()
            ax.plot(mean.index, mean.to_numpy(), color=color, linewidth=2, label=group)
            if column == "val_spearman":
                ax.annotate(
                    group, (mean.index[-1], mean.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#3d3d3a", va="center",
                )
        ax.set_title(title, fontsize=11, loc="left", color="#1f1f1e")
        ax.set_xlabel("epoch", color="#5f5e5a")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.tick_params(colors="#5f5e5a", labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Mean over seeds (thick), seeds (thin)", fontsize=10, color="#5f5e5a", x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("groups", nargs="+", help="<label>=<run id without -s<seed>>")
    parser.add_argument("--wandb-dir", type=Path, default=Path("runs/pathways/wandb"))
    parser.add_argument("--monitor", default="val/tcga-luad/median_spearman")
    parser.add_argument("--at-epochs", type=int, nargs="+", default=[0, 2, 5, 10])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames, labels = [], []
    for item in args.groups:
        label, sep, prefix = item.partition("=")
        if not sep:
            raise SystemExit(f"expected <label>=<run id prefix>, got {item!r}")
        run_ids = sorted(
            {m.group(1) for p in args.wandb_dir.glob(f"run-*-{prefix}-s*") if (m := re.search(rf"({re.escape(prefix)}-s\d+)$", p.name))}
        )
        if not run_ids:
            raise SystemExit(f"no W&B runs {prefix}-s* in {args.wandb_dir}")
        for run_id in run_ids:
            frames.append(fit_history(args.wandb_dir, run_id, args.monitor).assign(group=label, run=run_id))
        labels.append(label)
    curves = pd.concat(frames, ignore_index=True)[["group", "run", "epoch", "train_loss", "val_loss", "val_spearman"]]
    rho = curves["val_spearman"]
    if ((rho < -1) | (rho > 1)).any():
        raise ValueError("val Spearman outside [-1, 1]")
    if (curves[["train_loss", "val_loss"]] < 0).any().any():
        raise ValueError("negative loss")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(f"{args.output}.csv", index=False)
    summary = summarize(curves, args.at_epochs)
    command = "uv run python -m xaiwsi.evaluation.wandb_curves " + shlex.join(sys.argv[1:])
    text = (
        f"Learning curves from the local W&B files (monitor `{args.monitor}`; mean over seeds). "
        f"Generated by:\n\n```bash\n{command}\n```\n\n{markdown_table(summary)}\n\n"
        f"![learning curves]({Path(f'{args.output}.png').name})\n"
    )
    Path(f"{args.output}.md").write_text(text)
    plot(curves, labels, Path(f"{args.output}.png"))
    print(text)


if __name__ == "__main__":
    main()
