# Finetune BLOOM for NER

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

## General Information about BLOOM:

BLOOM uses a transformer architecture consisting of an input embedding layer, 70 transformer blocks, and an output language modeling layer, as shown in the figure below. Each transformer block has a self-observation layer and a multi-layer perceptron layer with norms for the input and post-observation layers.

- 176B parameters decoder-only architecture (GPT-like)
- 70 layers - 112 attention heads per layers - hidden dimensionality of 14336 - 2048 tokens sequence length

![](https://miro.medium.com/max/1400/1*uwWJBgEx3Rtovbcb7HcRdA.jpeg)

**Sources:**:

- Documentation: https://huggingface.co/docs/transformers/model_doc/bloom
- Model: https://huggingface.co/bigscience/bloom
- Github: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#readme

## Documentation as available in the Transformers Package on Huggingface.co:

- Tokenizer Class: https://huggingface.co/docs/transformers/glossary#attention-mask
- Trainer Class: https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/trainer#transformers.Trainer
- Finetuning using Trainer: https://huggingface.co/docs/transformers/training
- Token Classification: https://huggingface.co/docs/transformers/tasks/token_classification and https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb

## BLOOM Architecture explained:

- The Technology Behind BLOOM Training: https://huggingface.co/blog/bloom-megatron-deepspeed
- Understand BLOOM, the Largest Open-Access AI, and Run It on Your Local Computer:
  https://towardsdatascience.com/run-bloom-the-largest-open-access-ai-model-on-your-desktop-computer-f48e1e2a9a32

## Data used to train BLOOM:

- Multilingual: 46 languages: Full list is here: [https://bigscience.huggingface.co/blog/building-a-tb-scale-multilingual-dataset-for-language-modeling](https://bigscience.huggingface.co/blog/building-a-tb-scale-multilingual-dataset-for-language-modeling)
- 341.6 billion tokens (1.5 TB of text data)
- Tokenizer vocabulary: 250 680 tokens

![](https://github.com/bigscience-workshop/model_card/blob/main/assets/data/pie_v2.svg?raw=true)

**Sources:**:

- Corpus Map: https://huggingface.co/spaces/bigscience-catalogue-lm-data/corpus-map
- Building a TB Scale Multilingual Dataset for Language Modeling: https://bigscience.huggingface.co/blog/building-a-tb-scale-multilingual-dataset-for-language-modeling
