import pandas as pd
import lightkurve as lk
from lightkurve import TessLightCurve
import os

FITS_DIRECTORY = "fits_files"
TOI_DATA_PATH = "toi_data.csv"

if not os.path.exists(FITS_DIRECTORY):
    os.makedirs(FITS_DIRECTORY)

def download_fits(folder_path: str, tid: str, overwrite: bool =False) -> None:

    """
    Downloads the fits files for a given tid.
    Parameters:
        folder_path (str): path to the folder where the fits files should be downloaded
        tid (str): tid id of the fits file
        overwrite (bool, optional): overwrite existing files. Defaults to False.
    """
    try:

        search: lk.SearchResult = lk.search_lightcurve(f"TIC {tid}", mission="TESS", author="SPOC")
        if len(search) == 0:
            print(f"No fits files found for {tid}")
            return

        lc: TessLightCurve = search[0].download()

        lc.to_fits(
            path=f"{folder_path}/{tid}.fits",
            overwrite=overwrite
        )
        print(f"{tid}: download complete.")
    except Exception as err:
        print(f"Error processing {tid}: {err}")


def load_catalog(data: str) -> pd.DataFrame:

    """
    Loads the catalog from the fits files.

    Parameters:
        data (str): path to the fits file

    Returns:
        pd.DataFrame: catalog dataframe
    """

    data = pd.read_csv(data)

    df = data[["tid", "tfopwg_disp"]]

    clean_df = df[df["tfopwg_disp"].isin(["CP", "FP"])].copy()

    return clean_df

if __name__ == "__main__":

    catalog = load_catalog(data=TOI_DATA_PATH)

    count = 0
    for index, row in catalog.iterrows():

        tid = row["tid"]
        download_fits(
            folder_path=FITS_DIRECTORY,
            tid=tid,
            overwrite=True
        )
        count += 1
        print(f"{count}/{len(catalog)} complete.")