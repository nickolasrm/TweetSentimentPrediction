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

- [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

### 370k English words corpus

A words tag DataSet

#### Columns

- word: string containing a word
- pos_tag: string flags for each word category

| pos_tag    | Meaning                                              |
|-----------------|------------------------------------------------------|
| CC              | coordinating conjunction                             |
| CD              | cardinal digit                                       |
| DT              | determiner                                           |
| EX              | existential there                                    |
| FW              | foreign word                                         |
| IN              | preposition/subordinating conjunction                |
| JJ              | adjective (large)                                    |
| JJR             | adjective, comparative (larger)                      |
| JJS             | adjective, superlative (largest)                     |
| LS              | list item marker                                     |
| MD              | modal (could, will)                                  |
| NN              | noun, singular                                       |
| NNS             | noun plural                                          |
| NNP             | proper noun, singular                                |
| NNPS            | proper noun, plural                                  |
| PDT             | predeterminer                                        |
| POS             | possessive ending (parent\ 's)                       |
| PRP             | personal pronoun (hers, herself, him,himself)        |
| PRP dollar-sign | possessive pronoun (her, his, mine, my, our )        |
| RB              | adverb (occasionally, swiftly)                       |
| RBR             | adverb, comparative (greater)                        |
| RBS             | adverb, superlative (biggest)                        |
| RP              | particle (about)                                     |
| SYM             | symbol                                               |
| TO              | infinite marker (to)                                 |
| UH              | interjection (goodbye)                               |
| VB              | verb (ask)                                           |
| VBG             | verb gerund (judging)                                |
| VBD             | verb past tense (pleaded)                            |
| VBN             | verb past participle (reunified)                     |
| VBP             | verb, present tense not 3rd person singular(wrap)    |
| VBZ             | verb, present tense with 3rd person singular (bases) |
| WDT             | wh-determiner (that, what)                           |
| WP              | wh- pronoun (who)                                    |
| WP dollar-sign  | possessive wh-pronoun                                |
| WRB             | wh- adverb (how)                                     |

#### Links

- [Kaggle](https://www.kaggle.com/datasets/ruchi798/part-of-speech-tagging)

## Experiments

### Neural Network and dummies

#### Preprocess

1. Take all examples and balance the number of target classes in order to avoid predicting the most recurrent class
2. Lowercase texts
3. Split train and test in a 80:20 proportion

#### Feature Engineering

1. Convert text to dummies
2. Remove all mention words (starts with @)
3. Remove connectives
4. Apply PCA to reduce dimensionality

#### Training

1. Create a simple nn
2. Perform a GridSearch over the train data to find the optimal topology
3. Save the best network

#### Testing

1. Classify the train data

## Tools used

- Kedro: pipeline management framework
- Keras: deep learning framework
- Pandas: dataset handling lib
- Jupyter: Experiment testing

## Credits

Made with love by nickolasrm ❤️

## References


