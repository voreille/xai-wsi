import json

import pandas as pd

csv_path = "/mnt/nas6/data/CPTAC/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv"
output_json_path = (
    "/home/valentin/workspaces/xai-wsi/data/tiles/cptac/slide_filenames.json"
)

df = pd.read_csv(csv_path)

df_filtered = df[
    (df["Embedding_Medium"] == "FFPE") & (df["Specimen_Type"] == "tumor_tissue")
]

slide_ids = df_filtered["Slide_ID"].tolist()
slide_filenames = [f"{slide_id}.svs" for slide_id in slide_ids]
with open(output_json_path, "w") as f:
    json.dump(slide_filenames, f)
