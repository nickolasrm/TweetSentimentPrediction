# TweetSentimentPrediction
A set of experiments for predicting what sentiment a tweet express

## DataSets

### Sentiment140 dataset with 1.6 million tweets

This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

#### Columns

- target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- ids: The id of the tweet ( 2087)
- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- flag: The query (lyx). If there is no query, then this value is NO_QUERY.
- user: the user that tweeted (robotickilldozr)
- text: the text of the tweet (Lyx is cool)

#### References

- Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

#### Links

- [Sentiment140](http://help.sentiment140.com/)

## Download

This project automatically downloads the required datasets before any run.

![Download pipeline representation](/screenshots/download.jpg)

> Note: This only happens once. After it, the project will use caches

## Preprocess

![Data engineering pipeline representation](/screenshots/data_engineering.jpg)

### Sentiment140

1. Take all examples on `Sentiment140` and balance the number of target classes in order to avoid predicting the most recurrent class
2. Lowercase texts
3. Set header
4. Remove mentions (words starting with @)
5. Remove # (the symbol only)

## Experiments

### Naive Bayes (Multinomial)

![Naive Bayes pipeline representation](/screenshots/naive-bayes.jpg)

#### Feature Selection

1. Select text
2. Select target

#### Training

1. Split features and target into test and train data
2. Fit CountVectorizer removing stop words to get all words by frequency
3. Fit the words by frequency in a MultinomialNB
4. Wraps it up in a Pipeline

#### Testing

1. Classify the train data
2. Make a classification report

## Usage

To test this project by your own, do the following steps:

1. Clone the repo
2. Go to the project folder and run `pip install -r src/requirements.txt`
3. Optional: install `requirements.dev.txt` and `requirements.test.txt` in order to edit the project
4. Run `kedro run --pipeline <EXPERIMENT>` changing `<EXPERIMENT>` by the name of one of the folders under `src/tweets_sentiment_prediction/pipelines`

### Custom testing

1. Create a file `x_custom.csv` under `data/05_model_input` containing a `text` column and the tweets on its content
2. Run any pipeline with the `custom` tag by using the regular `kedro run --pipeline <PIPE> --tag custom`

## Tools used

- Kedro: pipeline management framework
- Sklearn: ml and statistic algorithms lib
- Pandas: dataset handling lib
- Jupyter: Experiment/hypothesis testing tool
- Mlflow: Experiment tracking tool

## Credits

Made with love by nickolasrm ❤️

## References

- [StatQuest with Josh Starmer - Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
- [Hardikkumar Dhaduk - Performing Sentiment Analysis With Naive Bayes Classifier!](https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/)