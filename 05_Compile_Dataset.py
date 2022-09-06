# %% [markdown]
# # Finetuning BLOOM for NER: Compile Corpus

# %% [markdown]
# ***

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
# ## Download Dataset for Finetuning

# %% [markdown]
# See:
# * Dataset on Huggingface: https://huggingface.co/datasets/wikiann
# * Load Datasets: https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/loading_methods

# %%
# Specify list of languages
languages = ["en","de", "fr", "es", "zh"]
#languages = ["en"]
dataset_name = "wikiann"

# Downloa first language
dataset_train = load_dataset(dataset_name, languages[0],  split="train")
dataset_valid = load_dataset(dataset_name, languages[0],  split="validation")
dataset_test =  load_dataset(dataset_name, languages[0],  split="test")
languages.pop(0)

# Merge with additional languages
for language in languages:
    
    print(f"Download Dataset for Language {language}")
    
    # Combine train splits
    dataset_train_new = load_dataset(dataset_name, language,  split="train")
    dataset_train = concatenate_datasets([dataset_train, dataset_train_new])

    # Combine validation splits
    dataset_valid_new = load_dataset(dataset_name, language,  split="validation")
    dataset_valid = concatenate_datasets([dataset_valid, dataset_valid_new])
    
    # Combine test splits
    dataset_test_new = load_dataset(dataset_name, language,  split="test")
    dataset_test = concatenate_datasets([dataset_test, dataset_test_new])

# %%
dataset = DatasetDict({
    "train":dataset_train,
    "test":dataset_test, 
    "validation":dataset_valid
    })

# %% [markdown]
# **Limit Dataset Size for Testing:**

# %%
## Sample a subset of datapoints
#num_samples = 1000
#sample_ids = list(range(0,num_samples))
#
## Reduce the size of the dataset
#dataset_train = dataset_train.select(sample_ids)
#dataset_valid = dataset_valid.select(sample_ids)
#dataset_test = dataset_test.select(sample_ids)
#
#print("Training Examples:", len(dataset_train))

# %% [markdown]
# **Save combined Dataset:**

# %%
data_path = "./data/dataset_multilingual.pkl"
with open(data_path, 'wb') as pickle_file:
    pickle.dump(obj = dataset, file=pickle_file)

# %% [markdown]
# ### About the Dataset:

# %% [markdown]
# **Splits:**

# %%
dataset

# %% [markdown]
# **Training Examples:**

# %%
print("Dataset Object Type:", type(dataset["train"]))
print("Training Examples:", len(dataset["train"]))

# %% [markdown]
# **Sample Structure:**

# %%
dataset["train"][95]

# %% [markdown]
# **Class Labels:**

# %%
label_list = dataset["train"].features[f"ner_tags"].feature.names
print(label_list)

# %%



