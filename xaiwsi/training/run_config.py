"""Read the `data` section of saved LightningCLI run configs.

The pathway CLI runs in data subclass mode, so newer run configs store the data module
arguments under `data.init_args` (with `data.class_path`); older ones store them flat.
"""

from __future__ import annotations

from typing import Any

from xaiwsi.data.cohorts import CohortSpec, ResolvedCohort


def data_args(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data") or {}
    return data.get("init_args", data)


def resolve_cohort(config: dict[str, Any], name: str) -> ResolvedCohort:
    """Paths of cohort `name` from a run config. Two-cohort configs (flat `tcga_*` /
    `cptac_*` keys) expose the cohorts `tcga` and `cptac`."""
    data = data_args(config)
    if "train_cohorts" in data:
        specs = [CohortSpec(**spec) for spec in data["train_cohorts"] + data["test_cohorts"]]
        processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    else:
        specs = [
            CohortSpec(
                name=prefix,
                rna_dir=data.get(f"{prefix}_rna_dir"),
                wsi_metadata=data.get(f"{prefix}_wsi_metadata"),
                embedding_dir=data.get(f"{prefix}_embedding_dir"),
                slides_root_dir=data.get(f"{prefix}_slides_root_dir"),
            )
            for prefix in ("tcga", "cptac")
            if f"{prefix}_embedding_dir" in data
        ]
        processed_root, embedding = None, ""
    for spec in specs:
        if spec.name == name:
            return spec.resolve(processed_root, embedding)
    raise KeyError(f"Cohort {name!r} not in the run config (have {[s.name for s in specs]}).")

