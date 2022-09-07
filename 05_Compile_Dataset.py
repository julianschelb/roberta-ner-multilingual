#!/usr/bin/env python
# coding: utf-8

# # Finetuning RoBERTa for NER: Compile Corpus

# ***

# ## Imports

# In[1]:


from transformers import (BertTokenizerFast,
                          RobertaTokenizerFast,
                          AutoTokenizer,
                          BertForTokenClassification,
                          RobertaForTokenClassification,
                          DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, 
                          TrainingArguments, Trainer)
from datasets import load_dataset, concatenate_datasets, DatasetDict
import pickle
import torch
import os


# ## Load Tokenizer

# **Load Model and Tokenizer:**
# 
# Information about model variants can be found here: https://huggingface.co/docs/transformers/model_doc/roberta

# In[2]:


model_name = "xlm-roberta-large" #"bert-base-multilingual-cased" #xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", add_prefix_space=True) #AutoTokenizer(use_fast = True)
#model = AutoModelForTokenClassification.from_pretrained(f"{model_name}")


# ## Download Dataset for Finetuning

# See:
# * Dataset on Huggingface: https://huggingface.co/datasets/wikiann
# * Load Datasets: https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/loading_methods

# In[3]:


# Specify list of languages
languages = ["en","de", "fr", "ne", "zh"]
#languages = ["en", "de"]
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


# In[4]:


dataset = DatasetDict({
    "train":dataset_train,
    "test":dataset_test, 
    "validation":dataset_valid
    })


# **Limit Dataset Size for Testing:**

# In[5]:


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


# **Save combined Dataset:**

# In[6]:


data_path = "./data/dataset_multilingual.pkl"
with open(data_path, 'wb') as pickle_file:
    pickle.dump(obj = dataset, file=pickle_file)


# ### About the Dataset:

# **Splits:**

# In[7]:


dataset


# **Training Examples:**

# In[8]:


print("Dataset Object Type:", type(dataset["train"]))
print("Training Examples:", len(dataset["train"]))


# **Sample Structure:**

# In[9]:


dataset["train"][95]


# **Class Labels:**

# In[10]:


label_list = dataset["train"].features[f"ner_tags"].feature.names
print(label_list)


# In[ ]:




