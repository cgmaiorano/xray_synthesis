from tcia_utils import nbia
import os
import pandas as pd

download_dir = "./LIDC-IDRI-subset"

series_list = nbia.getSeries(collection="LIDC-IDRI", modality="CT")
print(f"Found {len(series_list)} CT series")

series_df = pd.DataFrame(series_list)

os.makedirs(download_dir, exist_ok=True)

nbia.downloadSeries(
    series_data=series_df, number=200, path=download_dir, input_type="df"
)

print("Download complete.")
