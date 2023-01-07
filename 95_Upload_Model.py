# %% [markdown]
# # Finetuning RoBERTa for NER: Upload Model

# %% [markdown]
# https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForTokenClassification

# %% [markdown]
# ***

# %%
from transformers import (BertTokenizerFast,
                          RobertaTokenizerFast,
                          BertForTokenClassification,
                          RobertaForTokenClassification,
                          DataCollatorForTokenClassification,
                          AutoTokenizer, 
                          AutoModelForTokenClassification, 
                          TrainingArguments, Trainer)
from datasets import load_dataset, concatenate_datasets
import torch
import os

# %% [markdown]
# ## Load Model and Tokenizer

# %%
# model_name = "xlm-roberta-large" #"bert-base-multilingual-cased" #xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-final/", add_prefix_space=True) #AutoTokenizer(use_fast = True)
model = AutoModelForTokenClassification.from_pretrained("./results/checkpoint-final/")

# %%
model.config

# %% [markdown]
# ## Upload to Huggingface

# %%
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HF_ACCESS_TOKEN")

# %%
#from huggingface_hub import notebook_login
# notebook_login()
from huggingface_hub import login

# non-blocking login
login(token=token)

# blocking login: widget in a notebook or prompt in an interpreter
#login()

# %%
model.push_to_hub("roberta-ner-multilingual-test", use_temp_dir=True )

# %%
tokenizer.push_to_hub("roberta-ner-multilingual-test", use_temp_dir=True )

# %%



