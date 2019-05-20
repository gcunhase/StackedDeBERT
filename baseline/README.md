## Baseline models
* NLU Platforms (more information at Wiki)
  1. [Rasa](#1-rasa) (Open source)
  2. [IBM Watson](#2-ibm-watson)
  3. [Google DialogFlow](#3-google-dialogflow-apiai) (Api.ai)
  4. [SAP Conversational AI](#4-sap-recastai) (Recast.ai)
     * New intent classification algorithm as of April 2019 ([source](https://headwayapp.co/sapconversationalai-changelog))
  
* [Semantic Hashing with Classifier](#semantic-hashing-with-classifier)

## Dependencies
```
pip install -r requirements.txt
pip install tqdm boto3 requests matplotlib ftfy pandas unicodecsv pandas
```
> If using conda environment, change pip to conda or easy_install

## Semantic Hashing with Classifier
* Paper: *Subword Semantic Hashing for Intent Classification on Small Datasets* [[arXiv](https://arxiv.org/abs/1810.07150)] [[code](https://github.com/kumar-shridhar/Know-Your-Intent/blob/master/updated_semhash_pipeline.ipynb)]
* Requirements
  ```
  conda install -c conda-forge spacy
  python -m spacy download en_core_web_lg
  ```

* Obtain formatted dataset: `python sota_semantic_hashing/data_formatting.py`
* Run `python sota_semantic_hashing/main_original_paper.py` for each dataset and perc
* Obtain file with best and average accuracies for each dataset and perc: `python sota_semantic_hashing/get_best_sem_acc.py`
* P.S: Best out of all classifiers is stored in Wiki

## NLU Platforms
### 1. Rasa
* [Quickstart](https://rasa.com/docs/nlu/quickstart/)
* Same dataset format as LUIS (has to be version `2.x.x`)
* Requirements: `pip install rasa rasa_nlu sklearn_crfsuite tensorflow spacy`, `spacy download en`
* Pipeline: spacy, tensorflow
* Train with Rasa model and eval to obtain F1-score: `python rasa_train_eval.py`

### 2. IBM Watson
* [Source](https://console.bluemix.net/developer/watson/services): https://cloud.ibm.com
* Install `conda install -c conda-forge pandas_ml`
* Get appropriate train and test datasets in *.csv* format: `python watson/data_formatting.py`
* Create *Natural Language Classifier* service [here](https://console.bluemix.net/developer/watson/services)
    * Click `Launch Tool` after creating service
* Train ([reference code](https://github.com/watson-developer-cloud/python-sdk))
    * P.S.: Max of 8 train instances at once
    * Make `watson_params.json` with *url*, *api_key* and *train_csv_file*
    * Train: `python watson/language_classifier_train.py`
    * **Save classifier ID** (*nlc_id*) for each dataset (can be checked [here](https://dataplatform.cloud.ibm.com/))
* Test ([reference code](https://github.com/joe4k/wdcutils/blob/master/notebooks/NLCPerformanceEval.ipynb))
    * Wait until classifier's status is "Available" (check once in a while with `python watson/language_classifier_status.py`)
    * Complete `watson_params.json` with *nlc_id*, *test_csv_file*, *results_dir*, *results_csv_file*, and *confmatrix_csv_file*
    * Test: `python watson/language_classifier_test.py`

### 3. Google DialogFlow (Api.ai)
* [Intents doc](https://dialogflow.com/docs/intents)
* Use [Detect Intent API](github.com/googleapis/dialogflow-python-client-v2)):
 ```
 pip install dialogflow google-cloud google-cloud-storage google-api-core
 conda install -c mutirri google-api-python-client
 ```
* Create Agent at [cloud platform project](https://console.dialogflow.com/api-client) 
* Click on `Service Account` link and create key (download `json` file and save it in dialogflow folder)
* Add intents and train: `python dialogflow/train.py --dataset_name ChatbotCorpus --perc 0`
* Immediately go your agent's site and wait for a green pop up that says "Agent training complete"
* Evaluate: `python dialogflow/eval.py --dataset_name ChatbotCorpus --perc 0`

### 4. SAP (Recast.ai)
* Install `pip install requests sapcai`
* Create `new bot` [here](https://cai.tools.sap) or [here](https://cai.tools.sap/bot-connector)
* Manually add intents
* Create `csv` file by running `python sap/[train/test]_data_formatting.py`
* Manually upload `csv` file for each intent
* Test with `python sap/eval.py`

## Notes
Problem with encoding of German words in Chatbot Corpus. Solved with `ensure_ascii=False` in JSON dump.
```
with open(results_dir_path, 'w') as outfile:
    json.dump(data_dict, outfile, indent=2, ensure_ascii=False)
```
