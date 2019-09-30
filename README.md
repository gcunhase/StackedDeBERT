## About
* Task: Text classification from noisy data
    * Twitter Sentiment Classification
    * Incomplete Intent Classification

* Model: Stacked Denoising BERT 
  > BERT + DeBERT (Denoising AutoEncoder + BERT)

* Baseline models
  * [BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
  * NLU Platforms: [Rasa](https://rasa.com) (spacy and tf), [Dialogflow](https://dialogflow.com), and [SAP Conversational AI](https://cai.tools.sap) 
  * [Semantic Hashing with Classifier](https://github.com/kumar-shridhar/Know-Your-Intent)

## Dependencies
Python 3.6 (3.7.3 tested), PyTorch 1.0.1.post2, CUDA 9.0 or 10.1
```
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

## How to Use
### 1. Dataset
* NLU Benchmark datasets with missing data (Check [Dataset README](./data/README.md))
* Training done on Incomplete + Complete Data

### 2. Pre-fine-tune BERT
* Twitter Sentiment Corpus
```
CUDA_VISIBLE_DEVICES=0,1,2 ./scripts/twitter_sentiment/run_bert_classifier_inc_with_corr.sh
```
> Script for *Inc+Corr* dataset. Scripts corresponding to *Inc* and *Corr* are also available in the same folder.

* Incomplete Intent Corpus
```
CUDA_VISIBLE_DEVICES=0,1,2 ./scripts/snips_intent/run_bert_classifier_comp_inc_nomissingtag.sh
```
> Script for noisy data (*Comp Inc*). Script for clean, non-noisy data, is also available (*Complete*).

### 3. Train Stacked DeBERT
* Training on Twitter Corpus
```
CUDA_VISIBLE_DEVICES=0,1,2 ./scripts/twitter_sentiment/run_stacked_debert_dae_classifier_twitter_inc_with_corr.sh
```
> Make sure the OUTPUT directory is the same as the fine-tuned BERT or copy the BERT model to your new output dir.

* Training on NLU Evaluation Corpora for percentage of missing words 0.1-0.8 and autoencoder epochs 100-5000.
```
CUDA_VISIBLE_DEVICES=0,1,2 ./scripts/snips_intent/run_stacked_debert_dae_classifier_comp_inc_nomissingtag.sh
```
> Make sure the OUTPUT directory is the same as the fine-tuned BERT or copy the BERT model to your new output dir.

## Acknowledgment
* Dataset: [Snips](https://github.com/snipsco/nlu-benchmark)
* HuggingFace's [BERT PyTorch code](https://github.com/huggingface/pytorch-pretrained-BERT)

## Credits
In case you wish to use this code, please credit by citing it:

```
```

Email for further requests or questions: `gwena.cs@gmail.com`