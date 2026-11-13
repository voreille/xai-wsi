"""One-off move to the data / models / outputs layout of 2026-11-13 (see xaiwsi/paths.py).

data/processed/<cohort>/ keeps data only (metadata, rna, rna-embeddings, tissue, wsi-embeddings,
wsi-tiles); fitted models go to models/, slide features computed with them to outputs/.
Only renames (same filesystem); nothing is deleted except empty folders. Refuses to overwrite.
Paths stored in the moved JSON files (sae.json, train.json, prototypes.json) are rewritten.

    uv run python scripts/migrate_layout.py            # print the plan
    uv run python scripts/migrate_layout.py --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
P = DATA / "processed"
MODELS = REPO / "models"
OUTPUTS = REPO / "outputs"
GLOBAL = {"sae", "panther", "tile-pca", "tissue"}
CONCH = "conch-v1.5-fp16"
# Old naming: CONCH runs carried the store as a prefix; the store is now a path level.
STRIP = f"{CONCH}_"


class Plan:
    def __init__(self) -> None:
        self.moves: list[tuple[Path, Path]] = []
        self.rmdirs: list[Path] = []

    def move(self, src: Path, dst: Path) -> None:
        if src.exists():
            self.moves.append((src, dst))

    def check(self) -> None:
        dsts = [d for _, d in self.moves]
        if len(set(dsts)) != len(dsts):
            raise SystemExit("two sources map to the same destination")
        clash = [d for d in dsts if d.exists()]
        if clash:
            raise SystemExit(f"destinations exist already, e.g. {clash[:3]}")

    def apply(self) -> None:
        for src, dst in self.moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        for d in self.rmdirs:
            if d.exists():
                remove_empty(d)


def remove_empty(d: Path) -> None:
    for sub in sorted(d.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if sub.is_dir():
            sub.rmdir()  # fails loudly if not empty
        elif sub.name == ".embedding_store.json":  # descriptor of a store that never got data
            sub.unlink()
        else:
            raise SystemExit(f"not empty: {sub}")
    d.rmdir()


def bag_store(bag: Path) -> str:
    """Tile store a concept bag was computed on (from its sae.json)."""
    meta = json.loads((bag / "sae.json").read_text())
    return Path(meta["embedding_dir"]).name


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def build() -> Plan:
    plan = Plan()
    cohorts = sorted(d for d in P.iterdir() if d.is_dir() and d.name not in GLOBAL)
    for c in cohorts:
        if c.name in {"ignite", "anorak"}:
            continue
        # tiles: one folder per grid
        if (c / "wsi-tiles" / "patches_json").exists():
            plan.move(c / "wsi-tiles", c / "wsi-tiles" / "224px")  # via a temporary name, see main
        plan.move(c / "wsi-tiles-512", c / "wsi-tiles" / "512px")
        # tissue probes and tile QC: tissue/<store>/<probe>
        for probe in ("ignite", "anorak"):
            src = c / "tissue" / probe
            if src.exists() and (src / "probe.json").exists():
                store = Path(json.loads((src / "probe.json").read_text())["embedding_dir"]).name
                plan.move(src, c / "tissue" / store / probe)
        if (c / "tile-qc").exists():
            for store_dir in sorted((c / "tile-qc").iterdir()):
                plan.move(store_dir, c / "tissue" / store_dir.name / "tile-qc")
            plan.rmdirs.append(c / "tile-qc")
        # slide encoder output next to the tile stores; its check goes to outputs
        if (c / "titan").exists():
            plan.move(c / "titan" / "slides.parquet", c / "wsi-embeddings" / "titan" / "slides.parquet")
            plan.move(c / "titan" / "reference_cosine.csv", OUTPUTS / "conch_titan" / "titan_check" / f"{c.name}_reference_cosine.csv")
            plan.rmdirs.append(c / "titan")
        # concept bags and caches -> outputs
        if (c / "concepts").exists():
            for item in sorted((c / "concepts").iterdir()):
                if item.is_dir():
                    store = bag_store(item)
                    name = item.name.removeprefix(f"{store}_")
                    plan.move(item, OUTPUTS / "concept-bags" / store / name / c.name)
                else:
                    plan.move(item, OUTPUTS / "cache" / c.name / item.name)
            plan.rmdirs.append(c / "concepts")
        if (c / "panther").exists():
            for item in sorted((c / "panther").iterdir()):
                meta = json.loads((item / "prototypes.json").read_text())
                store = Path(meta["prototypes"]).parent.name
                plan.move(item, OUTPUTS / "panther-bags" / store / item.name / c.name)
            plan.rmdirs.append(c / "panther")

    # fitted models
    for store_dir in sorted((P / "sae").iterdir()) if (P / "sae").exists() else []:
        for run in sorted(store_dir.iterdir()):
            plan.move(run, MODELS / "sae" / store_dir.name / run.name.removeprefix(f"{store_dir.name}_"))
    for kind in ("panther", "tile-pca"):
        for store_dir in sorted((P / kind).iterdir()) if (P / kind).exists() else []:
            plan.move(store_dir, MODELS / kind / store_dir.name)
    for kind in ("sae", "panther", "tile-pca"):
        plan.rmdirs.append(P / kind)
    # reference tile sets of the tissue probes are datasets
    for probe_dir in sorted((P / "tissue").iterdir()) if (P / "tissue").exists() else []:
        for store_dir in sorted(probe_dir.iterdir()):
            plan.move(store_dir, P / probe_dir.name / "wsi-embeddings" / store_dir.name)
    plan.rmdirs.append(P / "tissue")

    # legacy tiling folders of the first pipeline
    plan.move(DATA / "tiles" / "cptac" / "slide_filenames.json", P / "cptac-luad" / "metadata" / "slide_filenames.json")
    plan.move(DATA / "tiles" / "tcga", P / "tcga-luad" / "wsi-tiles" / "224px")
    plan.move(DATA / "tiles" / "cptac", P / "cptac-luad" / "wsi-tiles" / "224px")
    plan.rmdirs += [DATA / "tiles", DATA / "embeddings"]

    # CONCH probe runs named with the old prefix: rename; their probes are rerun by the sweep
    for sweep in sorted(OUTPUTS.glob("conch_titan/sweep_*")):
        for run in sorted(sweep.glob(f"{STRIP}*")):
            plan.move(run, sweep / run.name.removeprefix(STRIP))
    return plan


def rewrite_json() -> list[str]:
    changed = []
    patterns = [
        ("data/processed/sae/", "models/sae/"),
        ("data/processed/panther/", "models/panther/"),
        ("data/processed/tile-pca/", "models/tile-pca/"),
        (f"/{CONCH}/{STRIP}", f"/{CONCH}/"),
    ]
    files = [*MODELS.glob("sae/*/*/*.json"), *MODELS.glob("panther/*/*/*.json")]
    files += [*OUTPUTS.glob("concept-bags/*/*/*/sae.json"), *OUTPUTS.glob("panther-bags/*/*/*/prototypes.json")]
    for f in files:
        text = f.read_text()
        new = text
        for old, rep in patterns:
            new = new.replace(old, rep)
        if new != text:
            f.write_text(new)
            changed.append(rel(f))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build()
    plan.check()
    for src, dst in plan.moves:
        print(f"mv {rel(src)} -> {rel(dst)}")
    for d in plan.rmdirs:
        if d.exists():
            print(f"rmdir {rel(d)}")
    print(f"{len(plan.moves)} moves")
    if not args.apply:
        print("dry run; pass --apply")
        return
    # A cohort's own wsi-tiles store becomes wsi-tiles/224px: move it aside first.
    final = []
    for src, dst in plan.moves:
        if dst == src / "224px":
            tmp = src.with_name(src.name + "-224-tmp")
            src.rename(tmp)
            final.append((tmp, dst))
        else:
            final.append((src, dst))
    plan.moves = final
    plan.apply()
    changed = rewrite_json()
    print(f"applied; rewrote paths in {len(changed)} JSON files")


if __name__ == "__main__":
    main()
