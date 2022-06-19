from typing import Dict
import pandas as pd


def _select_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the tweets from preprocessed `sentiment140`

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame: containing `text` only
    """
    return df[['text']]


def _select_target(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts target from preprocessed `sentiment140`

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame: containing `target` only
    """
    return df[['target']]


def select_tweets_features(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extracts only the useful features

    Args:
        df (pd.DataFrame)

    Returns:
        Dict[str, pd.DataFrame]
    """
    return {'tweets': _select_text_column(df), 'target': _select_target(df)}
