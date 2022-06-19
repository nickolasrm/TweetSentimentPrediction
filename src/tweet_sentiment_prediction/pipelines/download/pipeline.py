import os
from pathlib import Path
from kedro.pipeline import Pipeline, node, pipeline

from tweet_sentiment_prediction.pipelines.download.nodes import (
    download_sentiment140
)


def _is_datasets_downloaded() -> bool:
    """Checks if all required datasets were already downloaded

    Returns:
        bool
    """
    return all([
        os.path.isfile(str(Path('data') / '01_raw' / filename))
        for filename in ['raw_tweets.parquet']
    ])


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_sentiment140,
            inputs=[],
            outputs='raw_tweets',
            name='download_sentiment140'
        )
    ] if not _is_datasets_downloaded() else [],
                    tags='download')
