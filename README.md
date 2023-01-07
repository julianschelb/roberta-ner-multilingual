# Finetune RoBERTa for NER

Finetuning RoBERTa for multilingual Named-entity recognition.

## Setup

First, create a python environment. We will use [pyenv](https://github.com/pyenv/pyenv), but other options will likely work to.

Use th following commands to (1) install a specific python version, (2) create a new virtual environment, (3) activate that environment and (4) install python dependencies.

```bash
pyenv install -v 3.10.8
pyenv virtualenv 3.10.8 finetune-transformer
pyenv activate finetune-transformer
pip install -r requirements.txt
```

## Run

Run a notebook headless:
```bash
pyenv activate finetune-transformer
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute original.ipynb
```

Execute a python file:
```bash
pyenv activate finetune-transformer
nohup python 05_Compile_Dataset.py &
```

## General Information about RoBERTa:

Liu et al. presented an improved BERT variant named RoBERTa. To improve BERT, the authors conducted a series of experiments investigating the impact of training data and training parameters on the downstream performance of BERT. The authors determined that training BERT with a larger batch size and using larger input sequences during pretraining increases downstream performance. Furthermore, during pretraining, the authors forwent the sentence prediction task and used a dynamic method to mask tokens of the input task during the Mask Language phase. In the original work, Devlin et al. statically masked tokens before training the model.
Before feeding an input sequence into the model, tokenization needs to be applied. The original BERT paper uses BytePair Encoding (BPE). The input sequence is split up into mixed pieces representing words or only characters. In comparison to a wordlevel-only approach, this enables the representation of a more diverse dataset, which is especially beneficial when training on a multilingual dataset. However, using BPE, the vocabulary size snowballs. Radfort et al. proposed an even more universal and more efficient approach. By splitting up the input sequence on byte level instead per Unicode character as the smallest unit, a universal vocabulary allows for representing any input text with a modest size of 50k units. Unicode-based approaches typically result in vocabulary sized between 10k-100k subword units.

The overall improved performance makes RoBERTa, in many cases, the obvious choice over the original BERT models. Also, the universal input encoding makes RoBERTa more convenient to use in a multilingual setting.

See:

- Auto Tokenizer: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
- Auto Model for Token Classification: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
- RoBERTa on Huggingface: https://huggingface.co/docs/transformers/model_doc/roberta
- XLM-RoBERTA-large on Huggingface: https://huggingface.co/xlm-roberta-large
- BERT in Huggingface: https://huggingface.co/docs/transformers/model_doc/bert
- BERT multilingual on Huggingface: https://huggingface.co/bert-base-multilingual-cased

## Dataset:

The complete WikiANN dataset includes training examples for 282 languages and was constructed from Wikipedia. Training examples are extracted in an automated manner, exploiting entities mentioned in Wikipedia articles, often are formatted as hy- perlinks to the source article. Provided NER tags are in the IOB2 format. Named entities are classified as location (LOC), person (PER), or organization (ORG). 

See:

- Wikiann: https://huggingface.co/datasets/wikiann

## Other:

- Add special Tokens: https://github.com/huggingface/transformers/issues/5232, https://github.com/huggingface/tokenizers/issues/247

## Related Papers

* Pan, X., Zhang, B., May, J., Nothman, J., Knight, K., & Ji, H. (2017). Cross-lingual Name Tagging and Linking for 282 Languages. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1946–1958). Association for Computational Linguistics.
* Rahimi, A., Li, Y., & Cohn, T. (2019). Massively Multilingual Transfer for NER. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 151–164). Association for Computational Linguistics.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V.. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. 




