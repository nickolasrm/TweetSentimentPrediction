"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from tweet_sentiment_prediction.pipelines.data_engineering\
    .preprocess import preprocess_tweets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_tweets,
            inputs=[
                'raw_tweets', 'params:remove_mentions',
                'params:convert_hashtags'
            ],
            outputs='preprocess_tweets',
            name='preprocess_tweets'
        ),
    ],
                    tags='data_engineering')
