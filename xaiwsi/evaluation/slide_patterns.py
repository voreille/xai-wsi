"""Slide-level predominant growth pattern from the tile embeddings (e.g. DHMC LUAD), and whether tile layout helps.

A cohort with one predominant-pattern label per slide and no RNA (DHMC:
``xaiwsi.preprocessing.dhmc.metadata``) is used as a test set for pattern information in the
slides. Labels are only used for evaluation, except in the cross-validated diagnostic.

1. Zero-shot from the applied ANORAK tile probe (``xaiwsi.tissue.apply``), no fit on the cohort:
   - ``soft``: mean pattern fractions over all tiles, renormalized over the patterns;
   - ``vote``: share of tumour tiles (IGNITE tumour fraction >= ``--tumour-threshold``) whose
     most likely pattern is each pattern;
   - ``vote-ring<r>``: same after averaging each tumour tile's pattern fractions with the
     tumour tiles within Chebyshev distance r on the tile grid (``tile_grid``);
   - ``vote-ring<r>-shuffled``: same smoothing after permuting the tiles over the tumour-tile
     positions of their slide (``--shuffles`` permutations, metrics averaged): same amount of
     averaging, no layout. ``ring`` minus ``shuffled`` is the effect of the spatial layout.
   ANORAK's ``cribriform`` counts as ``acinar`` (a WHO acinar variant; DHMC has no cribriform
   label). Per variant: accuracy and balanced accuracy of the largest share, and per pattern
   the one-vs-rest AUC of its share (independent of calibration).
2. ``l2s``: a ridge on slide means fit on the TCGA training split (``--config``, ``--gene-sets``,
   as ``xaiwsi.evaluation.cohort_probe``) predicts the Tavernari solid-minus-lepidic axes:
   AUC solid vs rest and Spearman with the pattern grade.
3. Diagnostic, fit on the labels (repeated stratified ``--folds``-fold CV): balanced logistic
   regression on the slide means, on the soft shares, and on the ``vote-ring`` shares.

    uv run python -m xaiwsi.evaluation.slide_patterns \\
        --embedding-dir data/processed/dhmc-luad/wsi-embeddings/h-optimus-1-fp16 \\
        --labels data/processed/dhmc-luad/metadata/growth_patterns.csv \\
        --config configs/training/pathway_cohorts.yaml \\
        --gene-sets data/external/tavernari/derived/tavernari_l2s_approx.json \\
        --output-dir outputs/slide_patterns/dhmc-luad

Writes ``slides.parquet`` (per slide: label, shares of every variant, L2S predictions),
``metrics.csv`` (one row per method and metric), ``bootstrap.csv`` (paired bootstrap over
slides of ring vs shuffled and vote vs soft) and ``metrics.md``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.data.pathway import tile_grid
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.geneset_vs_hallmarks import add_axis_scores
from xaiwsi.evaluation.learning_curve import make_probe, slide_means
from xaiwsi.evaluation.pattern_hallmarks import DEFAULT_PATTERN_GRADE, parse_pattern_grade
from xaiwsi.rna.pathways.tavernari import load_tavernari_signatures
from xaiwsi.tissue.features import tissue_dir
from xaiwsi.training.run_config import data_args

# ANORAK class -> label pattern. Cribriform is an acinar variant.
PATTERN_OF = {
    "cribriform": "acinar",
    "acinar": "acinar",
    "lepidic": "lepidic",
    "micropapillary": "micropapillary",
    "papillary": "papillary",
    "solid": "solid",
}
TUMOUR_CLASSES = ["Tumor epithelium", "Keratinization"]


def probe_fractions(root: Path, name: str, slide_id: str) -> tuple[np.ndarray, list[str]]:
    classes = json.loads((root / name / "probe.json").read_text())["classes"]
    return np.load(root / name / "tiles" / f"{slide_id}.npy").astype(np.float32), classes


def ring_mean(values: np.ndarray, grid: np.ndarray, radius: int) -> np.ndarray:
    """Mean of ``values`` (N x K) over the tiles within Chebyshev ``radius`` on the grid, self included."""
    offsets = grid - grid.min(axis=0)
    height, width = offsets[:, 1].max() + 1 + 2 * radius, offsets[:, 0].max() + 1 + 2 * radius
    total = np.zeros((height, width, values.shape[1]), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    rows, cols = offsets[:, 1] + radius, offsets[:, 0] + radius
    np.add.at(total, (rows, cols), values)
    np.add.at(count, (rows, cols), 1.0)
    # Box sums via 2-D cumulative sums.
    def box(a: np.ndarray) -> np.ndarray:
        c = np.pad(a, [(1, 0), (1, 0)] + [(0, 0)] * (a.ndim - 2)).cumsum(0).cumsum(1)
        k = 2 * radius + 1
        out = np.zeros_like(a)
        out[radius : height - radius, radius : width - radius] = (
            c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
        )
        return out

    sums, counts = box(total), box(count)
    return (sums[rows, cols] / counts[rows, cols, None]).astype(np.float32)


def vote_shares(pattern_fractions: np.ndarray, patterns: list[str]) -> np.ndarray:
    """Share of tiles whose most likely ANORAK class maps to each label pattern."""
    winners = np.asarray(list(PATTERN_OF.values()))[pattern_fractions.argmax(axis=1)]
    return np.array([(winners == p).mean() if len(winners) else np.nan for p in patterns])


def merge_patterns(fractions: np.ndarray, classes: list[str], patterns: list[str]) -> np.ndarray:
    """ANORAK class fractions (N x K) -> label-pattern fractions (N x P), cribriform into acinar."""
    out = np.zeros((len(fractions), len(patterns)), dtype=np.float32)
    for name, target in PATTERN_OF.items():
        out[:, patterns.index(target)] += fractions[:, classes.index(name)]
    return out


def classification_rows(method: str, shares: pd.DataFrame, labels: pd.Series, patterns: list[str]) -> list[dict]:
    predicted = shares[patterns].idxmax(axis=1)
    rows = [
        {"method": method, "metric": "accuracy", "value": float((predicted == labels).mean())},
        {"method": method, "metric": "balanced_accuracy", "value": balanced_accuracy_score(labels, predicted)},
    ]
    for p in patterns:
        rows.append({"method": method, "metric": f"auc_{p}", "value": roc_auc_score(labels == p, shares[p])})
    rows.append(
        {"method": method, "metric": "macro_auc", "value": float(np.mean([r["value"] for r in rows[2:]]))}
    )
    return rows


def paired_bootstrap(
    a: pd.DataFrame, b: pd.DataFrame, labels: pd.Series, patterns: list[str], n: int, seed: int
) -> dict:
    """Macro AUC and balanced accuracy of a minus b, resampling slides (both on the same draw)."""
    rng = np.random.default_rng(seed)
    y = labels.to_numpy()
    deltas = {"macro_auc": [], "balanced_accuracy": []}

    def scores(s: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
        aucs = [roc_auc_score(y[idx] == p, s[idx, k]) for k, p in enumerate(patterns) if 0 < (y[idx] == p).sum() < len(idx)]
        predicted = np.asarray(patterns)[s[idx].argmax(axis=1)]
        return float(np.mean(aucs)), balanced_accuracy_score(y[idx], predicted)

    sa, sb = a[patterns].to_numpy(), b[patterns].to_numpy()
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        (auc_a, bal_a), (auc_b, bal_b) = scores(sa, idx), scores(sb, idx)
        deltas["macro_auc"].append(auc_a - auc_b)
        deltas["balanced_accuracy"].append(bal_a - bal_b)
    (auc_a, bal_a), (auc_b, bal_b) = scores(sa, np.arange(len(y))), scores(sb, np.arange(len(y)))
    point = {"macro_auc": auc_a - auc_b, "balanced_accuracy": bal_a - bal_b}
    return {
        metric: (point[metric], *np.percentile(values, [2.5, 97.5])) for metric, values in deltas.items()
    }


def l2s_predictions(config: Path, gene_sets_path: Path, means: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """TCGA-train ridge on slide means -> Tavernari L2S axes on the given slides."""
    data = data_args(yaml.safe_load(config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    gene_sets = load_tavernari_signatures(gene_sets_path)
    names = list(gene_sets)
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, _ = select_cohorts(train_specs, test_specs, [train_specs[0].name])
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_used],
        [],
        gene_sets,
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )
    part = splits["train"][0]
    slide_ids = part.pairs["slide_id"].astype(str).tolist()
    x = slide_means(part.store, slide_ids, cache_dir / f"slide_means_{part.cohort}_{embedding}.parquet")
    y = part.scores.loc[part.pairs["sample_id"].astype(str), names].to_numpy()
    model = make_probe().fit(x.to_numpy(), y)
    predicted = add_axis_scores(pd.DataFrame(model.predict(means.to_numpy()), columns=names, index=means.index))
    print(f"L2S ridge: fit on {part.cohort} train (n = {len(y)})")
    return predicted[[c for c in predicted.columns if c.startswith("L2S_")]]


def cv_rows(name: str, x: np.ndarray, labels: pd.Series, patterns: list[str], folds: int, repeats: int) -> list[dict]:
    y = labels.to_numpy()
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
    bal, aucs = [], []
    proba = np.zeros((repeats, len(y), len(patterns)))
    for k, (train, test) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=3, class_weight="balanced", max_iter=5000, scoring="balanced_accuracy"),
        ).fit(x[train], y[train])
        order = [list(model.classes_).index(p) for p in patterns]
        proba[k // folds, test] = model.predict_proba(x[test])[:, order]
    for r in range(repeats):
        predicted = np.asarray(patterns)[proba[r].argmax(axis=1)]
        bal.append(balanced_accuracy_score(y, predicted))
        aucs.append(np.mean([roc_auc_score(y == p, proba[r][:, i]) for i, p in enumerate(patterns)]))
    return [
        {"method": f"cv:{name}", "metric": "balanced_accuracy", "value": float(np.mean(bal)), "sd": float(np.std(bal))},
        {"method": f"cv:{name}", "metric": "macro_auc", "value": float(np.mean(aucs)), "sd": float(np.std(aucs))},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="CSV with case_id and the label column")
    parser.add_argument("--label-column", default="dominant_pattern")
    parser.add_argument("--slides", type=Path, default=None, help="CSV slide_id,case_id (default: slide_id = case_id)")
    parser.add_argument("--tumour-threshold", type=float, default=0.5)
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--config", type=Path, help="CohortPathwayDataModule config for the TCGA L2S ridge")
    parser.add_argument("--gene-sets", type=Path, help="Tavernari .json/.gmt (with --config)")
    parser.add_argument(
        "--train-means-dir", type=Path, default=Path("outputs/cohort_probe"),
        help="where the training cohort's slide means are cached (as in cohort_probe)",
    )
    parser.add_argument("--pattern-grade", default=DEFAULT_PATTERN_GRADE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels).set_index("case_id")[args.label_column]
    if args.slides is not None:
        slide_case = pd.read_csv(args.slides).set_index("slide_id")["case_id"]
    else:
        slide_case = pd.Series(labels.index, index=labels.index)
    store = build_embedding_store_from_dir(root_dir=args.embedding_dir)
    root = tissue_dir(args.embedding_dir)
    slide_ids = sorted(set(store.slide_ids()) & set(slide_case.index))
    patterns = sorted(set(PATTERN_OF.values()))
    unknown = set(labels.loc[slide_case.loc[slide_ids]]) - set(patterns)
    if unknown:
        raise SystemExit(f"labels outside the ANORAK patterns: {sorted(unknown)}")

    records = {}
    variants = ["soft", "vote"] + [f"vote-ring{r}" for r in args.radii] + [f"vote-ring{r}-shuffled" for r in args.radii]
    shuffled = {r: [] for r in args.radii}
    rng = np.random.default_rng(0)
    for slide_id in tqdm(slide_ids, desc="slides"):
        _, coords, attrs = store.load(slide_id)
        grid = tile_grid(np.asarray(coords), attrs)
        anorak, anorak_classes = probe_fractions(root, "anorak", slide_id)
        ignite, ignite_classes = probe_fractions(root, "ignite", slide_id)
        if not len(anorak) == len(ignite) == len(grid):
            raise ValueError(f"{slide_id}: {len(anorak)} anorak, {len(ignite)} ignite rows, {len(grid)} tiles")
        merged = merge_patterns(anorak, anorak_classes, patterns)
        tumour = ignite[:, [ignite_classes.index(c) for c in TUMOUR_CLASSES]].sum(axis=1) >= args.tumour_threshold
        soft = merged.mean(axis=0)
        row = {"slide_id": slide_id, "n_tiles": len(grid), "n_tumour": int(tumour.sum())}
        row |= {f"soft:{p}": v for p, v in zip(patterns, soft / soft.sum(), strict=True)}
        # Vote on the ANORAK pattern classes only (background excluded), tumour tiles.
        pattern_idx = [anorak_classes.index(c) for c in PATTERN_OF]
        tiles, tiles_grid = anorak[tumour][:, pattern_idx], grid[tumour]
        row |= {f"vote:{p}": v for p, v in zip(patterns, vote_shares(tiles, patterns), strict=True)}
        for r in args.radii:
            smoothed = ring_mean(tiles, tiles_grid, r) if len(tiles) else tiles
            row |= {f"vote-ring{r}:{p}": v for p, v in zip(patterns, vote_shares(smoothed, patterns), strict=True)}
            draws = []
            for _ in range(args.shuffles):
                permuted = tiles[rng.permutation(len(tiles))]
                smoothed = ring_mean(permuted, tiles_grid, r) if len(tiles) else tiles
                draws.append(vote_shares(smoothed, patterns))
            shuffled[r].append(np.stack(draws))
            row |= {f"vote-ring{r}-shuffled:{p}": v for p, v in zip(patterns, np.mean(draws, axis=0), strict=True)}
        records[slide_id] = row
    slides = pd.DataFrame.from_dict(records, orient="index").drop(columns="slide_id")
    slides.index.name = "slide_id"
    slides.insert(0, "label", labels.loc[slide_case.loc[slides.index]].to_numpy())
    if (slides["n_tumour"] == 0).any():
        empty = slides.index[slides["n_tumour"] == 0].tolist()
        print(f"warning: no tumour tiles on {empty}; their vote shares are NaN and set to the soft shares")
        for v in variants[1:]:
            for p in patterns:
                slides.loc[empty, f"{v}:{p}"] = slides.loc[empty, f"soft:{p}"]
    y = slides["label"]

    def shares(variant: str) -> pd.DataFrame:
        return slides[[f"{variant}:{p}" for p in patterns]].set_axis(patterns, axis=1)

    rows = []
    for variant in variants:
        if variant.endswith("-shuffled"):
            # Metrics per shuffle draw, then averaged (not the metrics of the averaged shares).
            r = int(variant.removeprefix("vote-ring").removesuffix("-shuffled"))
            per_draw = [
                pd.DataFrame(classification_rows(variant, pd.DataFrame(draws, index=slides.index, columns=patterns), y, patterns))
                for draws in np.stack(shuffled[r], axis=1)
            ]
            stacked = pd.concat(per_draw)
            rows += stacked.groupby(["method", "metric"], sort=False)["value"].agg(["mean", "std"]).reset_index().rename(
                columns={"mean": "value", "std": "sd"}
            ).to_dict("records")
        else:
            rows += classification_rows(variant, shares(variant), y, patterns)

    boot = []
    pairs = [(f"vote-ring{r}", f"vote-ring{r}-shuffled") for r in args.radii] + [("vote", "soft")]
    pairs += [(f"vote-ring{r}", "vote") for r in args.radii]
    for a, b in pairs:
        for metric, (point, low, high) in paired_bootstrap(shares(a), shares(b), y, patterns, args.bootstrap, 0).items():
            boot.append({"a": a, "b": b, "metric": metric, "delta": point, "ci_low": low, "ci_high": high})
    boot = pd.DataFrame(boot)

    means = slide_means(store, list(slides.index), args.output_dir / "slide_means.parquet")
    if args.config is not None:
        if args.gene_sets is None:
            raise SystemExit("--config needs --gene-sets")
        l2s = l2s_predictions(args.config, args.gene_sets, means, args.train_means_dir)
        slides = slides.join(l2s)
        grade = parse_pattern_grade(args.pattern_grade)
        for column in l2s.columns:
            rows.append({"method": f"l2s:{column}", "metric": "auc_solid", "value": roc_auc_score(y == "solid", slides[column])})
            rows.append(
                {"method": f"l2s:{column}", "metric": "grade_spearman", "value": spearmanr(y.map(grade), slides[column]).statistic}
            )

    rows += cv_rows("slide-mean", means.loc[slides.index].to_numpy(), y, patterns, args.folds, args.repeats)
    rows += cv_rows("soft", shares("soft").to_numpy(), y, patterns, args.folds, args.repeats)
    for r in args.radii[:1]:
        rows += cv_rows(f"vote-ring{r}", shares(f"vote-ring{r}").to_numpy(), y, patterns, args.folds, args.repeats)

    metrics = pd.DataFrame(rows)
    bounded = metrics["metric"] != "grade_spearman"
    if not metrics.loc[bounded, "value"].between(0, 1).all():
        raise ValueError(f"metric outside [0, 1]:\n{metrics[bounded & ~metrics['value'].between(0, 1)]}")
    if not metrics.loc[~bounded, "value"].between(-1, 1).all():
        raise ValueError("Spearman outside [-1, 1]")
    if not boot[["delta", "ci_low", "ci_high"]].abs().le(1).all().all():
        raise ValueError("bootstrap delta outside [-1, 1]")

    slides.to_parquet(args.output_dir / "slides.parquet")
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    boot.to_csv(args.output_dir / "bootstrap.csv", index=False)

    counts = y.value_counts()
    wide = metrics[~metrics["method"].str.startswith(("l2s:", "cv:"))].pivot(index="method", columns="metric", values="value")
    wide = wide.loc[variants, ["balanced_accuracy", "accuracy", "macro_auc"] + [f"auc_{p}" for p in patterns]]
    sd = metrics.dropna(subset=["sd"]) if "sd" in metrics else metrics.iloc[0:0]
    other = metrics[metrics["method"].str.startswith(("l2s:", "cv:"))]
    command = "uv run python -m xaiwsi.evaluation.slide_patterns " + shlex.join(sys.argv[1:])
    text = (
        f"Predominant pattern on {len(slides)} slides ({', '.join(f'{p} {n}' for p, n in counts.items())}); "
        f"tumour tiles: IGNITE tumour fraction >= {args.tumour_threshold}; shuffled = mean over "
        f"{args.shuffles} permutations. Generated by:\n\n```bash\n{command}\n```\n\n"
        f"Zero-shot from the ANORAK tile probe (largest share; AUC of each pattern's share, one vs rest):\n\n"
        f"{markdown_table(wide.reset_index())}\n\n"
        f"Paired bootstrap over slides ({args.bootstrap} draws), a minus b:\n\n{markdown_table(boot)}\n\n"
        f"TCGA-trained L2S ridge and cross-validated diagnostics (sd over CV repeats):\n\n"
        f"{markdown_table(other)}\n"
    )
    if len(sd):
        text += f"\nShuffled variants, sd over permutations:\n\n{markdown_table(sd[~sd['method'].str.startswith('cv:')])}\n"
    (args.output_dir / "metrics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
