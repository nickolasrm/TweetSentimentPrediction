import re
import numpy as np
import pandas as pd

from tweet_sentiment_prediction.utils.pipeline import lowercase_dataset


def _set_tweets_header(df: pd.DataFrame) -> pd.DataFrame:
    """Defines raw `sentiment140` DataFrame header (raw doesn't have one)

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    return df


def _remove_mentions(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Remove all words starting with @

    Args:
        df (pd.DataFrame)
        text_col (str, optional). Defaults to 'text'.

    Returns:
        pd.DataFrame
    """
    regex = re.compile(r'@[A-Za-z0-9_]+')
    return df.assign(
        **{text_col: df[text_col].apply(lambda x: regex.sub('', x))}
    )


def _convert_hashtags(
    df: pd.DataFrame, text_col: str = 'text'
) -> pd.DataFrame:
    """Removes #

    Args:
        df (pd.DataFrame)
        text_col (str, optional). Defaults to 'text'.

    Returns:
        pd.DataFrame
    """
    return df.assign(
        **{text_col: df[text_col].apply(lambda x: x.replace('#', ''))}
    )


def _normalize_tweets_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms 0 or 4 to 0 or 1 classes

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame

    Note:
        this was done because the raw DataSet contains only 0 and 4 (negative
        or positive) classes, although its description says the DataSet may
        contain the class 2 (neutral)
    """
    return df.assign(target=np.where(df['target'] == 0, 0, 1))


def preprocess_tweets(
    df: pd.DataFrame,
    remove_mentions_: bool = True,
    convert_hashtags_: bool = True
) -> pd.DataFrame:
    """Does all preprocessing over the `sentiment140` DataSet

    Args:
        df (pd.DataFrame)
        remove_mentions_ (bool, optional). Defaults to True.
        convert_hashtags_ (bool, optional). Defaults to True.

    Returns:
        pd.DataFrame
    """
    df = (
        df.pipe(_set_tweets_header).pipe(lowercase_dataset
                                         ).pipe(_normalize_tweets_classes)
    )
    if remove_mentions_:
        df = _remove_mentions(df, text_col='text')
    if convert_hashtags_:
        df = _convert_hashtags(df, text_col='text')
    return df
