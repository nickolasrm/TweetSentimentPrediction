import os
from pathlib import Path
import gdown
import pandas as pd


def download_sentiment140() -> pd.DataFrame:
    """Downloads `sentiment140` dataset from gdrive

    Returns:
        pd.DataFrame
    """
    folder = Path('data') / '01_raw'

    zip_path = str(folder / 'sentiment140.zip')
    gdown.download(
        url=('https://drive.google.com/u/0/'
             'uc?id=0B04GJPshIjmPRnZManQwWEdTZjg'),
        output=zip_path,)
    gdown.extractall(path=zip_path, to=str(folder))

    os.remove(str(folder / 'testdata.manual.2009.06.14.csv'))
    os.remove(zip_path)

    file_path = str(folder / 'training.1600000.processed.noemoticon.csv')
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    os.remove(file_path)

    return df
