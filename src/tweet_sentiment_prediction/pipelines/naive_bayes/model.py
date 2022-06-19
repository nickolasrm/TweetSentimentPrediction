import logging
from typing import Tuple

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pprint

logger = logging.getLogger(__name__)


def split_train_test(
    text: pd.DataFrame, target: pd.DataFrame, train_pct: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits tweets and target into train and test by a percentage of train

    Args:
        text (pd.DataFrame)
        target (pd.DataFrame)
        train_pct (float)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            x_test, x_train, y_test, y_train
    """
    return train_test_split(text, target, test_size=train_pct)


def train_model(
    x: pd.DataFrame, y: pd.DataFrame, count_vectorizer_params: dict,
    naive_bayes_params: dict
) -> Pipeline:
    """Creates a pipeline for naive bayes and fits it using train data

    Args:
        x (pd.DataFrame)
        y (pd.DataFrame)
        count_vectorizer_params (dict)
        naive_bayes_params (dict)

    Returns:
        Pipeline
    """
    return Pipeline([
        ('count_vectorizer', CountVectorizer(**count_vectorizer_params)),
        ('naive_bayes', MultinomialNB(**naive_bayes_params)),
    ],
                    verbose=True).fit(x['text'], y['target'])


def test_model(model: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    """Predicts a tweets DataFrame given a model

    Args:
        model (Pipeline)
        x (pd.DataFrame)

    Returns:
        pd.DataFrame: containing yhat only
    """
    return pd.DataFrame({'yhat': model.predict(x['text'])})


def make_report(y: pd.DataFrame, yhat: pd.DataFrame) -> dict:
    """Makes a sklearn classification report

    Args:
        y (pd.DataFrame)
        yhat (pd.DataFrame)

    Returns:
        str
    """
    report = classification_report(
        y['target'],
        yhat['yhat'],
        output_dict=True,
        target_names=['negative', 'positive']
    )
    logger.info(f'\n\n{pprint.pformat(report)}')

    mlflow_report = {}
    for class_, subdict in report.items():
        if isinstance(subdict, dict):
            for k, v in subdict.items():
                mlflow_report[f'{class_}_{k}'] = {'value': v, 'step': 1}
        else:
            mlflow_report[class_] = {'value': subdict, 'step': 1}

    return mlflow_report
