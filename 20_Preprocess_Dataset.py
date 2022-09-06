# %% [markdown]
# # Finetuning BLOOM for NER: Preprocess Corpus
#  

# %% [markdown]
# ## Imports

# %%
from transformers import (BloomTokenizerFast,
                          BloomForTokenClassification,
                          DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, 
                          TrainingArguments, Trainer)
from datasets import load_dataset, concatenate_datasets, DatasetDict
import pickle
import torch
import os

# %% [markdown]
# ## Load Tokenizer

# %% [markdown]
# The list of available Models can be found here: https://huggingface.co/docs/transformers/model_doc/bloom

# %%
model_name = "bloom-560m"
tokenizer = BloomTokenizerFast.from_pretrained(f"bigscience/{model_name}", add_prefix_space=True)
#model = BloomForTokenClassification.from_pretrained(f"bigscience/{model_name}")

# %% [markdown]
# ## Load Dataset

# %%
data_path = "./data/dataset_multilingual.pkl"
with open(data_path, 'rb') as pickle_file:
    dataset = pickle.load(file=pickle_file)

# %% [markdown]
# ## Tokenize Dataset

# %% [markdown]
# ### Tokenize a Single Sample:

# %%
example = dataset["train"][50]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)

# %% [markdown]
# Sample after Tokenization:

# %%
tokenized_input

# %% [markdown]
# Word IDs:

# %%
tokenized_input.word_ids()

# %% [markdown]
# ### Tokenize Whole Dataset

# %%
def tokenizeInputs(inputs):
    
    tokenized_inputs = tokenizer(inputs["tokens"], max_length = 2048, truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    ner_tags = inputs["ner_tags"]
    labels = [ner_tags[word_id] for word_id in word_ids]
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs

# %%
example = dataset["train"][100]
tokenizeInputs(example)

# %%
tokenized_dataset = dataset.map(tokenizeInputs)

# %% [markdown]
# **Shuffle Dataset:**

# %%
tokenized_dataset = tokenized_dataset.shuffle()

# %% [markdown]
# **Count of Tokens in the Training Set:**

# %%
token_count = 0
for sample in tokenized_dataset["train"]:
    token_count = token_count + len(sample["labels"])
    
print("Tokens in Training Set:", token_count)

# %% [markdown]
# **Remove unnecessary columns:**

# %%
tokenized_dataset = tokenized_dataset.remove_columns(["tokens", "ner_tags", "langs", "spans"])

# %% [markdown]
# **Save processed Dataset:**

# %%
data_path = "./data/dataset_processed.pkl"
with open(data_path, 'wb') as pickle_file:
    pickle.dump(obj = dataset, file=pickle_file)


