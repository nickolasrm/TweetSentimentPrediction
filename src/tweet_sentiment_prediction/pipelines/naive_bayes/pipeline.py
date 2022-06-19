"""
This is a boilerplate pipeline 'naive_bayes'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from tweet_sentiment_prediction.pipelines.naive_bayes.model import (
    make_report, split_train_test, test_model, train_model)
from tweet_sentiment_prediction.pipelines.naive_bayes\
     .feature_selection import select_tweets_features
from tweet_sentiment_prediction.utils.pipeline import custom_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return (pipeline([
        node(func=select_tweets_features,
             inputs='preprocess_tweets',
             outputs={
                 'tweets': 'tweets',
                 'target': 'target'
             },
             name='select_features'),
        node(func=split_train_test,
             inputs=['tweets', 'target', 'params:train_pct'],
             outputs=['x_test', 'x_train', 'y_test', 'y_train'],
             name='split_train_test'),
        node(func=train_model,
             inputs=[
                 'x_train', 'y_train', 'params:count_vectorizer',
                 'params:naive_bayes'
             ],
             outputs='naive_bayes_model',
             name='train_naive_bayes'),
        node(func=test_model,
             inputs=['naive_bayes_model', 'x_test'],
             outputs='naive_bayes_prediction',
             name='test_naive_bayes',
             tags='predict'),
        node(func=make_report,
             inputs=['y_test', 'naive_bayes_prediction'],
             outputs='naive_bayes_report',
             tags='predict',
             name='naive_bayes_classification_report'),
    ],
                     tags='model') +
            custom_pipeline(func=test_model, model='naive_bayes_model'))
