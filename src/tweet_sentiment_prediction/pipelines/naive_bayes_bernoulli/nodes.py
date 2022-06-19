import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import HashingVectorizer


def train_model(
    x: pd.DataFrame,
    y: pd.DataFrame,
    hashing_vectorizer_params: dict,
    naive_bayes_params: dict
) -> Pipeline:
    """Creates a pipeline for naive bayes bernoulli and fits it
    using train data

    Args:
        x (pd.DataFrame)
        y (pd.DataFrame)
        hashing_vectorizer_params (dict)
        naive_bayes_params (dict)

    Returns:
        Pipeline
    """
    return Pipeline([
        ('hash_vectorizer', HashingVectorizer(**hashing_vectorizer_params)),
        ('naive_bayes', BernoulliNB(**naive_bayes_params)),
    ],
                    verbose=True).fit(x['text'], y['target'])