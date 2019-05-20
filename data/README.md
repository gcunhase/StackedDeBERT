## About
Incomplete Intent Classification Dataset

Made by deleting irrelevant words calculated with TF-IDF

## Contents
[Datasets](#datasets) • [TF-IDF](#tf-idf) • [Joint Comp/Inc](#joint-completeincomplete-data)

## Dependencies
Python 3.7.2, requests, numpy, nltk

## Datasets
Complete data is saved in the `./complete_data/` directory

[Snips](#snips-nlu-corpus) • [NLU Evaluation Corpora](#nlu-evaluation-corpora)
> Source: [Snips](https://github.com/snipsco/nlu-benchmark) and [NLU Evaluation Corpora](https://github.com/sebischair/NLU-Evaluation-Corpora)

### Snips NLU Corpus
  | Label  | Intent                                      | Train | Test | Train+ | Example |
  | ------ | ------------------------------------------- | ----- | ---- | ------ | ------- |
  | 0      | AddToPlaylist                               | 1942  | 100  | 2000   | *"Add a track to Jazzy Dinner"* |
  | 1      | BookRestaurant                              | 1973  | 100  | 2000   | *"I want to book a restaurant for six people in Wagstaff AK."* |
  | 2      | GetWeather                                  | 2000  | 100  | 2000   | *"Will the sun be out in 1 minute in Searcy, Uganda"* |
  | 3      | PlayMusic                                   | 2000  | 100  | 2000   | *"Play my Black Sabbath: The Dio Years playlist."* |
  | 4      | RateBook                                    | 1956  | 100  | 2000   | *"Rate the current novel 3 stars"* |
  | 5      | SearchCreativeWork<br>SearchItem            | 1954  | 100  | 2000   | *"Could you find the TV series The Approach"* |
  | 6      | SearchScreeningEvent<br>SearchMovieSchedule | 1959  | 100  | 2000   | *"I want to know if there are any movies playing in the area."* |     
  > 16K crowdsourced queries: 13784 train, 700 test

### NLU Evaluation Corpora
#### Chatbot Corpus
  | Label  | Intent              | Train | Test | Train+ | Example |
  | ------ | ------------------- | ----- | ---- | ------ | ------- |
  | 0      | Departure Time      | 43    | 35   |   57   | *"when is the next train in muncher freiheit?"* | |
  | 1      | Find Connection     | 57    | 71   |   57   | *"i want to go marienplatz"* | |
  > 206 questions from german chatbot: 100 train, 106 test
  
#### Ask Ubuntu Corpus
  | Label  | Intent                  | Train | Test | Train+ | Example |
  | ------ | ----------------------- | ----- | ---- | ------ | ------- |
  | 0      | Make Update             | 10    | 37   |   17   | *"Upgrading from 11.10 to 12.04"* |
  | 1      | Setup Printer           | 10    | 13   |   17   | *"How to install a Lexmark z600 series printer?"* |
  | 2      | Shutdown Computer       | 13    | 14   |   17   | *"shut down without extra question"* |
  | 3      | Software Recommendation | 17    | 40   |   17   | *"What software can I use to view epub documents?"* |
  | 4      | None                    | 3     | 5    |   17   | *"Is there a Document scanning and archiving software?"* |
  > 162 Q&A: 53 train, 109 test
  
#### Web Applications Corpus
  | Label  | Intent           | Train | Test | Train+ | Example |
  | ------ | ---------------- | ----- | ---- | ------ | ------- |
  | 0      | Change Password  | 2     | 6    |    7   | *"Gmail user set up through Google Apps can't change their password"* |
  | 1      | Delete Account   | 7     | 10   |    7   | *"How can I delete my Twitter account?"* |
  | 2      | Download Video   | 1     | 0    |    7   | *"How do I download a YouTube video?"* |
  | 3      | Export Data      | 2     | 3    |    7   | *"How can I backup my wordpress.com hosted blog?"* |
  | 4      | Filter Spam      | 6     | 14   |    7   | *"Correctly Identifying Spam Messages"* |
  | 5      | Find Alternative | 7     | 16   |    7   | *"Google search engine alternatives"* |
  | 6      | Sync Accounts    | 3     | 6    |    7   | *"How do I sync Google Calendar with my Outlook Calendar?"* |
  | 7      | None             | 2     | 4    |    7   | *"Embedding stop time in a YouTube video link"* |
  > 89 Q&A: 30 train, 59 test

## TF-IDF
### Quick code
Check `tfidf_test.py`

### Make dataset
```
python make_dataset_tfidf.py
```

* 30% missing words in each sentence

| Dataset    | Incomplete example | Missing words |
| ---------- | ------- | ------------------ |
| Ask Ubuntu | *"Is _ a Linux _ manager _ a _ drop-down tree view? (like finder _ OS X)"* | *there, file, with, proper, in* |
| Chatbot    | *"when does _ _ train leaves at garching?"* | *the next* |
| Web Apps   | *"How _ I auto-delete some spam _ Gmail?"* | *can from* |
| SNIPS      | *"what is the forecast _ here _ blizzard conditions _ _ pm"* | *for for at five* |

* 50% missing words in each sentence

| Dataset    | Incomplete example | Missing words |
| ---------- | ------- | ------------------ |
| Ask Ubuntu | *"Is _ a Linux _ _ _ a _ drop-down _ view? (like _ _ OS X)"* | *there file manager with proper tree finder in* |
| Chatbot    | *"_ does _ _ train leaves _ garching?"* | *when the next at* |
| Web Apps   | *"How _ I auto-delete _ _ _ Gmail?"* | *can some spam from* |
| SNIPS      | *"_ _ the forecast _ _ _ blizzard conditions _ _ pm"* | *what is for here for at five* |

## Joint Complete/Incomplete Data
* In order for the model to be robust to missing data it also needs to be trained on sentences with missing words.
* After making the incomplete dataset with POS-tag or TF-IDF, there are two options
   1. Make dataset with Complete and Incomplete Data
      ```
      python make_joint_comp_inc_data.py
      ```
   2. Make dataset with Incomplete Data (add target sentence to `tsv` file to train autoencoder)
      ```
      python add_target_to_inc_data.py
      ```

## Notes
Initially, we used the special character `<miss>` to substitute missing words. However we realized that it turned into 3 tokens during tokenization: `<` , `miss` and `>`. Because of that, we decided to change that character to `_` by using the following function: 
  ```
  python change_missing_words_tag.py
  ```

