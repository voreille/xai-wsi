# %%
import numpy as np
import pandas as pd

df = pd.read_parquet(
    "/home/valentin/workspaces/xai-wsi/data/processed/tcga-luad/rna/log2_tpm.parquet"
)

# %%
df.head()

# %%

df.shape
