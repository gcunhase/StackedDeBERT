## About
* Incomplete Intent Classification Dataset
* Twitter Sentiment Classification Dataset 

## Contents
[Incomplete Intent Data](#1-incomplete-intent-dataset) • [Twitter Sentiment Data](#2-twitter-sentiment-dataset) • [Joint Comp/Inc](#3-joint-completeincomplete-data)

## Dependencies
Python 3.7.2, requests, numpy, nltk

## 1. Incomplete Intent Dataset
Snips NLU Corpus: Download from [Intent Classifier repository](https://github.com/gcunhase/IntentClassifier)

* Make TF-IDF dataset
```
python make_dataset_tfidf.py
```

* Examples of sentences with missing words

| Missing rate| Incomplete example |
| ----------- | ------------------ |
| 0%          | *"Please help me search the TV series A Mouse Divided."* |
| 10%         | *"Please help _ search the TV series A Mouse Divided."* |
| 20%         | *"Please help _ search the TV series A _ Divided."* |
| 30%         | *"_ help _ search the TV series A _ Divided."* |
| 40%         | *"_ _ _ search the TV series A _ Divided."* |
| 50%         | *"_ _ _ _ the TV series A _ Divided."* |
| 80%         | *"_ _ _ _ _ _ _ A _ Divided."* |
> Underscore tag is there for clearer understanding of where the words are missing.

* Change tags 
```
python change_missing_words_tag.py
```

## 2. Twitter Sentiment Dataset
* Tweets have natural human error (noise)
* Correct sentences obtained with Amazon MTurk

## 3. Joint Complete/Incomplete Data
* In order for the model to be robust to missing data it also needs to be trained on sentences with missing words.
* After making the incomplete dataset, there are two options
   1. Make dataset with Complete and Incomplete Data
      ```
      python make_joint_comp_inc_data.py
      ```
   2. Make dataset with Incomplete Data (add target sentence to `tsv` file to train autoencoder)
      ```
      python add_target_to_inc_data.py
      ```