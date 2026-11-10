"""Zero-shot predominant pattern from trained tile-level runs: does a spatial model beat its controls?

Runs trained on TCGA to predict the approximate Tavernari lepidic/solid scores
(``CohortPathwayDataModule`` with ``gene_sets``, ``scripts/pathway_patterns.sh``) are applied to
a cohort with pattern labels and no RNA (DHMC). From each run's predicted ``SOLID_<src>`` and
``LEPIDIC_<src>``, the axis ``L2S_<src>`` = solid - lepidic is scored against the labels:

- ``auc_solid``: solid vs all other slides;
- ``auc_acinar_lepidic`` / ``auc_solid_acinar``: the two steps of the lepidic < acinar < solid
  order (higher L2S for the second pattern);
- ``grade_spearman``: Spearman with the pattern grade (``--pattern-grade``).

Each group (``<label>=<run dir glob>``, one run per seed) is scored per seed (mean and sd) and
on its seed-mean prediction. ``--pairs a:b`` gives paired bootstraps over cases of the
seed-mean predictions (a minus b), e.g. the NCA against its no-step and shuffled-layout
controls. ``--reference <label>=<slides.parquet>`` adds L2S columns predicted elsewhere (the
slide-mean ridge of ``xaiwsi.evaluation.slide_patterns``). Several slides of a case are
averaged (``--slides`` CSV slide_id,case_id; default slide_id = case_id).

    uv run python -m xaiwsi.evaluation.slide_pattern_runs \\
        --embedding-dir data/processed/dhmc-luad/wsi-embeddings/h-optimus-1-fp16 \\
        --labels data/processed/dhmc-luad/metadata/growth_patterns.csv \\
        --groups 'tilemean=runs/pathways/xai-wsi/coh-*tav-tilemean-luad-percohort-s*' \\
                 't16lp=runs/pathways/xai-wsi/coh-*tav-t16lp-luad-percohort-s*' \\
        --reference ridge=outputs/slide_patterns/dhmc-luad/slides.parquet \\
        --pairs tilemean:t16lp --output-dir outputs/pattern_runs/dhmc-luad

Writes ``predictions/<run>.parquet`` (per slide, cached), ``metrics.csv``, ``bootstrap.csv``
and ``metrics.md``.
"""

from __future__ import annotations

import argparse
import glob
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.pathway import tile_grid
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.geneset_vs_hallmarks import add_axis_scores
from xaiwsi.evaluation.pattern_hallmarks import DEFAULT_PATTERN_GRADE, parse_pattern_grade
from xaiwsi.evaluation.predict_run import best_checkpoint, load_predictor
from xaiwsi.rna.pathways.tavernari import load_tavernari_signatures
from xaiwsi.training.run_config import data_args


def predict_slides(run_dir: Path, store, slide_ids: list[str], cache: Path, device: torch.device) -> pd.DataFrame:
    """slide_id x target predictions of one run (best checkpoint), cached as parquet."""
    if cache.exists():
        cached = pd.read_parquet(cache)
        if set(slide_ids) <= set(cached.index):
            return cached.loc[slide_ids]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = data_args(config)
    if not data.get("gene_sets"):
        raise SystemExit(f"{run_dir}: not trained on gene_sets (Hallmark run?)")
    names = list(load_tavernari_signatures(data["gene_sets"]))
    predictor = load_predictor(config, best_checkpoint(run_dir), device)
    rows = {}
    for slide_id in tqdm(slide_ids, desc=run_dir.name, leave=False):
        embeddings, coords, attrs = store.load(slide_id)
        x = torch.from_numpy(np.asarray(embeddings)).float().to(device)[None]
        c = torch.from_numpy(tile_grid(np.asarray(coords), attrs)).to(device)[None]
        with torch.no_grad():
            rows[slide_id] = predictor(x, coords=c)["pred"][0].cpu().numpy()
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=names)
    frame.index.name = "slide_id"
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache)
    return frame


def axes(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = add_axis_scores(predictions)
    return scored[[c for c in scored.columns if c.startswith("L2S_")]]


def pair_auc(labels: np.ndarray, score: np.ndarray, low: str, high: str) -> float:
    keep = np.isin(labels, [low, high])
    return roc_auc_score(labels[keep] == high, score[keep])


def scores(labels: np.ndarray, grade: np.ndarray, values: np.ndarray) -> dict[str, float]:
    return {
        "auc_solid": roc_auc_score(labels == "solid", values),
        "auc_acinar_lepidic": pair_auc(labels, values, "lepidic", "acinar"),
        "auc_solid_acinar": pair_auc(labels, values, "acinar", "solid"),
        "grade_spearman": spearmanr(grade, values).statistic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="CSV with case_id and the label column")
    parser.add_argument("--label-column", default="dominant_pattern")
    parser.add_argument("--slides", type=Path, default=None, help="CSV slide_id,case_id (default: slide_id = case_id)")
    parser.add_argument("--groups", nargs="+", required=True, help="<label>=<run dir glob>")
    parser.add_argument("--reference", nargs="*", default=[], help="<label>=<parquet with slide_id index and L2S_* columns>")
    parser.add_argument("--pairs", nargs="*", default=[], help="<a>:<b>, bootstrap of a minus b")
    parser.add_argument("--pattern-grade", default=DEFAULT_PATTERN_GRADE)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    labels = pd.read_csv(args.labels).set_index("case_id")[args.label_column]
    if args.slides is not None:
        slide_case = pd.read_csv(args.slides).astype({"slide_id": str}).set_index("slide_id")["case_id"]
    else:
        slide_case = pd.Series(labels.index, index=labels.index.astype(str))
    store = build_embedding_store_from_dir(root_dir=args.embedding_dir)
    slide_case = slide_case[slide_case.index.isin(set(store.slide_ids())) & slide_case.isin(labels.index)]
    slide_ids = sorted(slide_case.index)
    cases = sorted(set(slide_case))
    y = labels.loc[cases].to_numpy()
    grade = labels.loc[cases].map(parse_pattern_grade(args.pattern_grade)).to_numpy()
    if np.isnan(grade.astype(float)).any():
        raise SystemExit("labels without a pattern grade")

    def per_case(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[slide_ids].groupby(slide_case.loc[slide_ids].to_numpy()).mean().loc[cases]

    # group -> list of per-seed case x L2S frames
    seeds: dict[str, list[pd.DataFrame]] = {}
    for item in args.groups:
        label, sep, pattern = item.partition("=")
        if not sep:
            raise SystemExit(f"expected <label>=<glob>, got {item!r}")
        run_dirs = sorted(Path(p) for p in glob.glob(pattern))
        if not run_dirs:
            raise SystemExit(f"no runs match {pattern}")
        seeds[label] = [
            per_case(axes(predict_slides(r, store, slide_ids, args.output_dir / "predictions" / f"{r.name}.parquet", device)))
            for r in run_dirs
        ]
        print(f"{label}: {[r.name for r in run_dirs]}")
    for item in args.reference:
        label, _, path = item.partition("=")
        frame = pd.read_parquet(path)
        seeds[label] = [per_case(frame[[c for c in frame.columns if c.startswith("L2S_")]])]

    rows = []
    for label, frames in seeds.items():
        mean = sum(frames) / len(frames)
        for axis in mean.columns:
            per_seed = pd.DataFrame([scores(y, grade, f[axis].to_numpy()) for f in frames])
            ensemble = scores(y, grade, mean[axis].to_numpy())
            for metric, value in ensemble.items():
                rows.append(
                    {
                        "group": label,
                        "n_runs": len(frames),
                        "axis": axis,
                        "metric": metric,
                        "seed_mean": per_seed[metric].mean(),
                        "seed_sd": per_seed[metric].std() if len(frames) > 1 else np.nan,
                        "ensemble": value,
                    }
                )
    metrics = pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(cases), len(cases)) for _ in range(args.bootstrap)]
    boot = []
    for pair in args.pairs:
        a, _, b = pair.partition(":")
        if a not in seeds or b not in seeds:
            print(f"skip pair {pair}: group missing")
            continue
        ma, mb = sum(seeds[a]) / len(seeds[a]), sum(seeds[b]) / len(seeds[b])
        for axis in ma.columns.intersection(mb.columns):
            va, vb = ma[axis].to_numpy(), mb[axis].to_numpy()
            point_a, point_b = scores(y, grade, va), scores(y, grade, vb)
            deltas = {m: [] for m in point_a}
            for idx in draws:
                sa, sb = scores(y[idx], grade[idx], va[idx]), scores(y[idx], grade[idx], vb[idx])
                for m in deltas:
                    deltas[m].append(sa[m] - sb[m])
            for m, values in deltas.items():
                low, high = np.nanpercentile(values, [2.5, 97.5])
                boot.append({"a": a, "b": b, "axis": axis, "metric": m, "delta": point_a[m] - point_b[m], "ci_low": low, "ci_high": high})
    boot = pd.DataFrame(boot, columns=["a", "b", "axis", "metric", "delta", "ci_low", "ci_high"])

    auc = metrics["metric"] != "grade_spearman"
    for column in ("seed_mean", "ensemble"):
        if not metrics.loc[auc, column].between(0, 1).all() or not metrics.loc[~auc, column].between(-1, 1).all():
            raise ValueError(f"{column}: AUC outside [0, 1] or Spearman outside [-1, 1]")
    if len(boot) and not boot[["delta", "ci_low", "ci_high"]].abs().le(2).all().all():
        raise ValueError("bootstrap delta out of range")

    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    boot.to_csv(args.output_dir / "bootstrap.csv", index=False)
    counts = labels.loc[cases].value_counts()
    wide = metrics.pivot_table(index=["axis", "group"], columns="metric", values="ensemble", sort=False)
    seed_wide = metrics.assign(
        value=metrics["seed_mean"].map("{:.3f}".format) + " ± " + metrics["seed_sd"].map("{:.3f}".format)
    ).pivot_table(index=["axis", "group"], columns="metric", values="value", aggfunc="first", sort=False)
    command = "uv run python -m xaiwsi.evaluation.slide_pattern_runs " + shlex.join(sys.argv[1:])
    text = (
        f"Predicted L2S (solid - lepidic) vs the predominant pattern, {len(cases)} cases "
        f"({', '.join(f'{p} {n}' for p, n in counts.items())}). Generated by:\n\n```bash\n{command}\n```\n\n"
        f"Seed-mean prediction:\n\n{markdown_table(wide.reset_index())}\n\n"
        f"Per seed (mean ± sd):\n\n{markdown_table(seed_wide.reset_index())}\n"
    )
    if len(boot):
        text += f"\nPaired bootstrap over cases ({args.bootstrap} draws, seed-mean predictions), a minus b:\n\n{markdown_table(boot)}\n"
    (args.output_dir / "metrics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
