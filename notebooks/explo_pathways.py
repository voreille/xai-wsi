# %%
import pandas as pd
import numpy as np

df = pd.read_parquet("/home/valentin/workspaces/xai-wsi/data/processed/tcga-luad/rna/pathways/hallmarks.parquet")

# %%
df.shape
df.head()

# %%
values = df["value"].to_numpy()
values.mean()
# %%

import matplotlib.pyplot as plt

hallmark_cols = df.select_dtypes(include="number").columns
values = df[hallmark_cols].to_numpy().ravel()

plt.figure(figsize=(6, 4))
plt.hist(values, bins=50)
plt.xlabel("ssGSEA score")
plt.ylabel("Count")
plt.title("Distribution of TCGA Hallmark ssGSEA scores")
plt.tight_layout()
plt.show()
