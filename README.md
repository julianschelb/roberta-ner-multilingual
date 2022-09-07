# Finetune RoBERTA for NER

Finetuning BLOOM from Bigscience for multilingual Named-entity recognition.

## Setup

Install pyenv on linux:

```
apt install python3.10-venv
```

Setup virtual environment:

```
python3 -m venv ./venv
source venv/bin/activate # .\venv\Scripts\activate for Windows
```

Install Python dependencies:

```
pip install -r requirements.txt
```

## Run

```
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute original.ipynb
```

## General Information about RoBERTA:

- Auto Tokenizer: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
- Auto Model for Token Classification: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
- RoBERTa on Huggingface: https://huggingface.co/docs/transformers/model_doc/roberta
- XLM-RoBERTA-large on Huggingface: https://huggingface.co/xlm-roberta-large
- BERT in Huggingface: https://huggingface.co/docs/transformers/model_doc/bert
- BERT multilingual on Huggingface: https://huggingface.co/bert-base-multilingual-cased

## Datasets:

- Wikiann: https://huggingface.co/datasets/wikiann

## Other:

- Add special Tokens: https://github.com/huggingface/transformers/issues/5232
