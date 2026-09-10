import json
from pathlib import Path

import pandas as pd

PATHWAY_PATH = Path(
    "/home/valentin/workspaces/xai-wsi/data/processed/tcga-luad/rna/pathways/hallmarks.parquet"
)
CSV_OUTPUT_PATH = Path(
    "/home/valentin/workspaces/xai-wsi/data/processed/tcga-luad/rna/pathways/hallmarks.csv"
)

pathway_df = pd.read_parquet(PATHWAY_PATH)
pathway_df.to_csv(CSV_OUTPUT_PATH, index=True)
