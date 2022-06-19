"""Project pipelines."""
from functools import reduce
from typing import Dict

from kedro.pipeline import Pipeline
from tweet_sentiment_prediction.pipelines.download import pipeline as download
from tweet_sentiment_prediction.pipelines.data_engineering import (
    pipeline as data_engineering)
from tweet_sentiment_prediction.pipelines.naive_bayes import (pipeline as
                                                              naive_bayes)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipes = {'naive-bayes': naive_bayes.create_pipeline()}

    return {
        k:
        v + reduce(lambda a, b: a + b, [
            data_engineering.create_pipeline(),
            download.create_pipeline()
        ])
        for k, v in pipes.items()
    }
