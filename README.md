## About
* Task: Incomplete Intention Classification

* Model: Stacked Denoising BERT 
  > BERT + DeBERT (Denoising AutoEncoder + BERT)

* Baseline models
  * [BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
  * NLU Platforms: [Rasa](https://rasa.com) (spacy and tf), [Watson](https://cloud.ibm.com), [Dialogflow](https://dialogflow.com), and [SAP Conversational AI](https://cai.tools.sap) 
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

### 2. Pre-train BERT
Training on Chatbot Corpus, repeat for each available dataset
```
--task_name chatbot_intent --do_train --do_eval --do_lower_case --data_dir ./data/comp_with_incomplete_data_tfidf_lower_0.4/nlu_eval_chatbotcorpus/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir results_comp_inc/chatbot_ep3_bs4_0.4/
```

### 3. Train Stacked DeBERT
Training on NLU Evaluation Corpora for percentage of missing words 0.1-0.8 and autoencoder epochs 100-5000.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_stacked_debert_dae_classifier.sh
```
> Make sure the OUTPUT directory is the same as the trained BERT or copy the BERT model to your new output dir.

## Credits
In case you wish to use this code, please credit this repository or send me an email at `gwena.cs@gmail.com` with any requests or questions.
