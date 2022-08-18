# Indonesian BERT-CRF

This directory has code to train and evaluate BER-CRF based models on NER task using the SINGGALANG datasets.

## Environment Setup
The program code uses the python 3.6 environment and GPU. The following steps in using the python virtual environment are as follows:

### Update pip
```bash
py -m pip install --upgrade pip
```
or 
```bash
py -m pip –version
```
### Install virtual env
```bash
py -m pip install --user virtualenv
```
or 
```bash
py -m venv env env 
```
### Activate Virtual env
```bash
. \env\Scripts\activate
```
### Install Requirements 
```bash
pip install -r requirements.txt
```
### Run BERT-CRF
```bash
run_bert_harem.py \
    --bert_model bert-base-multilingual-cased \
    --labels_file data/classes-total.txt \
    --do_train \
    --train_file data/singgalang-train.json \
    --valid_file data/singgalang-dev.json \
    --num_train_epochs 10 \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_eval \
    --eval_file data/singgalang-test.json \
    --output_dir output
```
## Trained model
Here is checkpoints of trained NER models on ambigu and phrase dataset.

[bert-base-multilingual-cased](https://tinyurl.com/bert-base-multilingual-cased)
### Run BERT-CRF Ambigu
```bash
python run_bert_harem.py \
    --bert_model /bert-base-multilingual-cased/ \
    --labels_file data/classes-total.txt \
    --train_file data/singgalang-train.json \
    --valid_file data/singgalang-dev.json \
    --num_train_epochs 10 \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_eval \
    --eval_file data/singgalang-test_ambigu.json \
    --output_dir ambiguout/
```
### Run BERT-CRF Frasa
```bash
python run_bert_harem.py \
    --bert_model /bert-base-multilingual-cased/ \
    --labels_file data/classes-total.txt \
    --train_file data/singgalang-train.json \
    --valid_file data/singgalang-dev.json \
    --num_train_epochs 10 \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_eval \
    --eval_file data/singgalang-test_phrase.json \
    --output_dir frasaout/
```


