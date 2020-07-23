## Baseline models
* Know-Your-Intent
* NLU Services

## Dependencies
```
pip install -r requirements.txt
pip install tqdm boto3 requests matplotlib ftfy pandas unicodecsv pandas
```
> If `setuptools` error appears: `sudo apt-get install python-setuptools` 

> If using conda environment, change pip to conda or easy_install

## Know-Your-Intent
* Python 3.5/3.6 (otherwise there's a `expected unicode but received string` error)
* Paper: *Subword Semantic Hashing for Intent Classification on Small Datasets* [[arXiv](https://arxiv.org/abs/1810.07150)] [[code](https://github.com/kumar-shridhar/Know-Your-Intent/blob/master/semhash_pipeline.ipynb)]
* Requirements
```
conda install -c conda-forge spacy
python -m spacy download en en_core_web_lg
```

* Obtain formatted dataset: `python sota_semantic_hashing/data_formatting.py`
* Run `python sota_semantic_hashing/main_multiple_runs.py` for each dataset and perc
* Obtain file with best and average accuracies for each dataset and perc: `python sota_semantic_hashing/get_best_sem_acc.py`
* P.S: Best out of all classifiers is stored in Wiki

## NLU Platforms
### 1. Google DialogFlow (Api.ai)
* Use Detect Intent API:
 ```
 pip install dialogflow google-cloud google-cloud-storage google-api-core
 conda install -c mutirri google-api-python-client
 ```
* Create Agent at [cloud platform project](https://console.dialogflow.com/api-client) 
* Click on [`Service Account`](https://console.cloud.google.com/iam-admin/serviceaccounts?project=intent-stterror-mihupt) link and create key (download `json` file and save it in dialogflow folder)
* Add intents and train: `python dialogflow/train.py --dataset_name snips --perc 0`
* Immediately go your agent's site and wait for a green pop up that says "Agent training complete"
* Evaluate: `python dialogflow/eval.py --dataset_name snips --perc 0`

### 2. SAP (Recast.ai)
* Install `pip install requests sapcai`
* Create `new bot` at https://cai.tools.sap
* Manually add intents
* Create `csv` file by running `python sap/[train/test]_data_formatting.py`
* Manually upload `csv` file for each intent
* Test with `python sap/eval.py`

### 3. Rasa
* [Quickstart](https://rasa.com/docs/nlu/quickstart/)
* Same dataset format as LUIS (has to be version `2.x.x`)
* Requirements:
   * Python: `pip install rasa rasa_nlu sklearn_crfsuite tensorflow spacy`, `spacy download en`
   * Anaconda: `conda install -c ak_93 rasa-nlu`, `conda install -c conda-forge tensorflow spacy python-crfsuite scikit-learn nltk`, `spacy download en`, `conda install -c derickl sklean-crfsuite`
* Pipeline: spacy, tensorflow (1.13.1)
* Train with Rasa model and eval to obtain F1-score: `python rasa_train_eval.py`
