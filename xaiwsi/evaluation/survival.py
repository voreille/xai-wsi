"""Overall survival from slide features: per-feature Cox adjusted for stage and age, and C-index.

Cases are the RNA-paired cases of a ``CohortPathwayDataModule`` config split as everywhere else
(``build_pathway_splits``: TCGA train/val, test cohorts), restricted to cases with OS time > 0,
OS event, age and stage group in ``<cohort dir>/metadata/clinical.csv``. Stage enters as an
ordinal 1-4 (few stage IV cases in CPTAC). The RNA targets are not used.

Per-feature analysis (each table feature set, e.g. concept bags; columns active, > 0, on at
least ``--min-active`` of the training slides): ``OS ~ z(feature) + z(stage) + z(age)``, fit
separately on the training split and on each test cohort (z within that cohort), with BH-FDR on
the training split. Replication: of the features with training FDR < ``--fdr``, the share with
the same sign of log-HR on the test cohort and the share also at p < 0.05 there; plus the
Pearson correlation of the feature z-statistics between training and test over all features.

C-index arms: ``clinical`` (stage, age; unpenalised), ``<features>`` and
``<features>+clinical``. Features are prepared as in ``cohort_probe`` (table blocks clipped to
the training range), standardised, projected on the top ``--pcs`` training PCs, then a ridge Cox
whose penalizer is picked on the val split's C-index from ``--penalizers``. Harrell's C on val
and test with case bootstrap intervals (``--n-boot``), and paired deltas between arms on the
same resamples.

    uv run python -m xaiwsi.evaluation.survival --config configs/training/pathway_cohorts.yaml \\
        --condition tcga-luad --features slide-mean \\
        concepts=bags:tcga-luad+tcga-lusc_k1024_top64_s0/slides_mean \\
        --output-dir outputs/survival/luad_valpick

Writes ``per_feature.csv``, ``replication.csv``, ``cindex.csv``, ``cindex_deltas.csv`` and
``summary.md`` (tables with the generating command).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xaiwsi import paths
from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.cohort_probe import TablePrep, markdown_table
from xaiwsi.evaluation.learning_curve import slide_means
from xaiwsi.metrics.survival import concordance, cox_fit
from xaiwsi.paths import is_table_block, table_path
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args

STAGES = {"I": 1, "II": 2, "III": 3, "IV": 4}


def parse_feature(entry: str) -> tuple[str, str]:
    """``label=block`` or ``block``; blocks are ``slide-mean`` or a table block."""
    label, block = entry.split("=", 1) if "=" in entry.split(":", 1)[0] else (entry, entry)
    if block != "slide-mean" and not is_table_block(block):
        raise SystemExit(f"Unknown feature block {block!r}")
    return label, block


def bh_fdr(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def zscore(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    ref = x if ref is None else ref
    sd = ref.std(axis=0)
    return (x - ref.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def load_survival(cohort_dir: Path) -> pd.DataFrame:
    clinical = pd.read_csv(cohort_dir / "metadata" / "clinical.csv")
    clinical["stage_num"] = clinical["stage_group"].map(STAGES)
    keep = clinical[["os_time", "os_event", "age", "stage_num"]].notna().all(axis=1) & (clinical["os_time"] > 0)
    return clinical[keep].set_index("case_id")[["os_time", "os_event", "age", "stage_num"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", default="tcga-luad", help="training cohorts, comma-separated")
    parser.add_argument("--features", nargs="+", required=True, help="[label=]slide-mean | [label=]bags:... | table:...")
    parser.add_argument("--min-active", type=float, default=0.1)
    parser.add_argument("--fdr", type=float, default=0.1)
    parser.add_argument("--pcs", type=int, default=32)
    parser.add_argument("--penalizers", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = [parse_feature(f) for f in args.features]

    data = data_args(yaml.safe_load(args.config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, test_used = select_cohorts(train_specs, test_specs, args.condition.split(","))
    resolved = {c.name: c.resolve(processed_root, embedding) for c in train_used + test_used}
    splits = build_pathway_splits(
        [resolved[c.name] for c in train_used],
        [resolved[c.name] for c in test_used],
        load_hallmarks(data["hallmark_gmt"]),
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )

    # One frame per (split, cohort): survival columns plus slide_id, for paired cases with follow-up.
    parts = {}
    for split, split_parts in splits.items():
        for part in split_parts:
            surv = load_survival(paths.cohort_dir(resolved[part.cohort].embedding_dir))
            pairs = part.pairs.assign(case_id=part.pairs["case_id"].astype(str)).set_index("case_id")
            frame = pairs[["slide_id"]].join(surv, how="inner")
            frame["slide_id"] = frame["slide_id"].astype(str)
            parts[split, part.cohort] = (frame, resolved[part.cohort].embedding_dir, part.store)
            print(f"{split} {part.cohort}: {len(frame)} of {len(pairs)} cases with OS, {int(frame.os_event.sum())} events")
    (train_key,) = [k for k in parts if k[0] == "train"]
    val_keys = [k for k in parts if k[0] == "val"]
    test_keys = [k for k in parts if k[0] == "test"]

    def raw(block: str, key) -> np.ndarray:
        frame, embedding_dir, store = parts[key]
        ids = list(frame["slide_id"])
        if block == "slide-mean":
            cache = paths.cache_dir(embedding_dir) / f"slide_means_{embedding_dir.name}.parquet"
            return slide_means(store, ids, cache).loc[ids].to_numpy(np.float64)
        table = pd.read_parquet(table_path(block, embedding_dir))
        table.index = table.index.astype(str)
        return table.loc[ids].to_numpy(np.float64)

    def outcome(key):
        frame = parts[key][0]
        return frame["os_time"].to_numpy(float), frame["os_event"].to_numpy(bool)

    def clinical(key, ref_key=train_key) -> np.ndarray:
        cols = ["stage_num", "age"]
        return zscore(parts[key][0][cols].to_numpy(float), parts[ref_key][0][cols].to_numpy(float))

    # Per-feature Cox, adjusted for stage and age, on train and each test cohort.
    per_feature, replication = [], []
    for label, block in features:
        if block == "slide-mean":
            continue
        x_train = raw(block, train_key)
        active = (x_train > 0).mean(axis=0) >= args.min_active
        columns = np.flatnonzero(active)
        fits = {}
        for key in [train_key, *test_keys]:
            x = raw(block, key)[:, columns]
            time, event = outcome(key)
            cov = zscore(parts[key][0][["stage_num", "age"]].to_numpy(float))
            res = []
            for j in range(len(columns)):
                if x[:, j].std() == 0:  # e.g. never active on a test cohort: not testable there
                    res.append((np.nan,) * 4)
                    continue
                fit = cox_fit(np.column_stack([zscore(x[:, j]), cov]), time, event)
                res.append((fit.beta[0], fit.se[0], fit.z[0], fit.p[0]))
            fits[key] = np.array(res)
        q_train = bh_fdr(fits[train_key][:, 3])
        for key in [train_key, *test_keys]:
            for j, column in enumerate(columns):
                beta, se, z, p = fits[key][j]
                per_feature.append({"features": label, "split": key[0], "cohort": key[1], "column": int(column),
                                    "log_hr": beta, "se": se, "z": z, "p": p,
                                    "q_train": q_train[j]})
        hits = q_train < args.fdr
        for key in test_keys:
            # Hits that are constant on the test cohort count as not replicated.
            same = np.sign(fits[key][hits, 0]) == np.sign(fits[train_key][hits, 0])
            finite = np.isfinite(fits[key][:, 2])
            replication.append({
                "features": label, "cohort": key[1], "tested": len(columns),
                "constant_on_test": int((~finite).sum()), "train_fdr_hits": int(hits.sum()),
                "same_sign": same.mean() if hits.any() else np.nan,
                "same_sign_p05": (same & (fits[key][hits, 3] < 0.05)).mean() if hits.any() else np.nan,
                "z_corr_all": np.corrcoef(fits[train_key][finite, 2], fits[key][finite, 2])[0, 1],
            })
    per_feature = pd.DataFrame(per_feature)
    replication = pd.DataFrame(replication)
    per_feature.to_csv(args.output_dir / "per_feature.csv", index=False)
    replication.to_csv(args.output_dir / "replication.csv", index=False)

    # C-index arms. Risk scores per arm and evaluation key.
    eval_keys = val_keys + test_keys
    risks: dict[str, dict] = {}
    chosen = {}
    time_tr, event_tr = outcome(train_key)
    fit = cox_fit(clinical(train_key), time_tr, event_tr)
    risks["clinical"] = {k: fit.risk(clinical(k)) for k in eval_keys}
    chosen["clinical"] = 0.0
    for label, block in features:
        x_train = raw(block, train_key)
        prep = TablePrep(x_train, 0.0, "clip") if block != "slide-mean" else (lambda x: x)
        prepped = prep(x_train)
        mean, sd = prepped.mean(axis=0), prepped.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        _, _, vt = np.linalg.svd((prepped - mean) / sd, full_matrices=False)
        proj = vt[: args.pcs].T
        pc_sd = ((prepped - mean) / sd @ proj).std(axis=0)
        # Training-fitted map: prep, standardise, project, unit-variance PCs (on train).
        pcs = {k: ((prep(raw(block, k)) - mean) / sd @ proj) / pc_sd for k in [train_key, *eval_keys]}

        def design(key, with_clinical: bool) -> np.ndarray:
            return np.column_stack([pcs[key], clinical(key)]) if with_clinical else pcs[key]

        for arm, with_clinical in ((label, False), (f"{label}+clinical", True)):
            best = None
            for penalizer in args.penalizers:
                fit = cox_fit(design(train_key, with_clinical), time_tr, event_tr, penalizer=penalizer)
                c_val = np.mean([concordance(*outcome(k), fit.risk(design(k, with_clinical))) for k in val_keys])
                if best is None or c_val > best[0]:
                    best = (c_val, penalizer, fit)
            _, chosen[arm], fit = best
            risks[arm] = {k: fit.risk(design(k, with_clinical)) for k in eval_keys}

    rng = np.random.default_rng(args.seed)
    rows, deltas = [], []
    arms = list(risks)
    for key in eval_keys:
        time, event = outcome(key)
        boots = np.empty((args.n_boot, len(arms)))
        for b in range(args.n_boot):
            idx = rng.integers(0, len(time), len(time))
            boots[b] = [concordance(time[idx], event[idx], risks[a][key][idx]) for a in arms]
        for a_i, arm in enumerate(arms):
            c = concordance(time, event, risks[arm][key])
            if not 0 <= c <= 1:
                raise ValueError(f"C-index out of range: {arm} {key} {c}")
            lo, hi = np.percentile(boots[:, a_i], [2.5, 97.5])
            rows.append({"arm": arm, "penalizer": chosen[arm], "split": key[0], "cohort": key[1], "n": len(time),
                         "events": int(event.sum()), "c_index": c, "ci_low": lo, "ci_high": hi})
        pairs_to_compare = [("clinical", a) for a in arms if a != "clinical"]
        labels = [label for label, _ in features]
        pairs_to_compare += [(labels[0], other) for other in labels[1:]]
        pairs_to_compare += [(f"{labels[0]}+clinical", f"{other}+clinical") for other in labels[1:]]
        for base, cand in pairs_to_compare:
            i, j = arms.index(base), arms.index(cand)
            d = boots[:, j] - boots[:, i]
            point = concordance(time, event, risks[cand][key]) - concordance(time, event, risks[base][key])
            deltas.append({"baseline": base, "candidate": cand, "split": key[0], "cohort": key[1], "delta": point,
                           "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                           "p_delta_le_0": (d <= 0).mean()})
    cindex, deltas = pd.DataFrame(rows), pd.DataFrame(deltas)
    cindex.to_csv(args.output_dir / "cindex.csv", index=False)
    deltas.to_csv(args.output_dir / "cindex_deltas.csv", index=False)

    command = "uv run python -m xaiwsi.evaluation.survival " + shlex.join(sys.argv[1:])
    rounded = lambda df: df.round(3) if len(df) else df  # noqa: E731
    sections = [
        f"Overall survival from slide features ({args.n_boot} case bootstraps, seed {args.seed}). Generated by:\n\n"
        f"```bash\n{command}\n```",
        "C-index (penalizer picked on val; clinical = stage + age):\n\n" + markdown_table(rounded(cindex)),
        "Paired C-index deltas (candidate − baseline, same resamples):\n\n" + markdown_table(rounded(deltas)),
    ]
    if len(replication):
        sections.append(
            f"Per-feature Cox adjusted for stage and age (features active on ≥ {args.min_active:g} of training "
            f"slides); replication of training FDR < {args.fdr:g} hits:\n\n" + markdown_table(rounded(replication))
        )
    (args.output_dir / "summary.md").write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections[1:]))


if __name__ == "__main__":
    main()
