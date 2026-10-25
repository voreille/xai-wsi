"""Per-case prediction files: the input of ``xaiwsi.evaluation.paired_bootstrap``.

Written by ``cohort_probe`` (``predictions/<condition>__<normalization>.parquet``) and by the
test stage of ``PathwayTraining`` (``<run-dir>/test_predictions.parquet``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def prediction_frame(
    case_ids,
    cohort: str | Sequence[str],
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: list[str],
) -> pd.DataFrame:
    """One row per case: case_id, cohort, split, true__<pathway>..., pred__<pathway>...

    ``cohort`` is one name for all cases or one name per case."""
    cohorts = [cohort] * len(y_true) if isinstance(cohort, str) else list(cohort)
    frame = pd.DataFrame({"case_id": list(case_ids), "cohort": cohorts, "split": split})
    true = pd.DataFrame(y_true, columns=[f"true__{n}" for n in names])
    pred = pd.DataFrame(y_pred, columns=[f"pred__{n}" for n in names])
    return pd.concat([frame, true, pred], axis=1)
