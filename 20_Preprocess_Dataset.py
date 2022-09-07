#!/usr/bin/env python
# coding: utf-8

# # Finetuning RoBERTa for NER: Preprocess Corpus
#  

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


# ## Load Dataset

# In[3]:


data_path = "./data/dataset_multilingual.pkl"
with open(data_path, 'rb') as pickle_file:
    dataset = pickle.load(file=pickle_file)


# ## Tokenize Dataset

# ### Tokenize a Single Sample:

# In[4]:


example = dataset["train"][50]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True,add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)


# Sample after Tokenization:

# In[5]:


tokenized_input


# Word IDs:

# In[6]:


tokenized_input.word_ids()


# ### Tokenize Whole Dataset

# In[7]:


def tokenizeInputs(inputs):
    
    tokenized_inputs = tokenizer(inputs["tokens"], max_length = 512, truncation=True, is_split_into_words=True, add_special_tokens=False)
    word_ids = tokenized_inputs.word_ids()
    ner_tags = inputs["ner_tags"]
    labels = [ner_tags[word_id] for word_id in word_ids]
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs


# In[8]:


example = dataset["train"][100]
tokenizeInputs(example)


# In[9]:


tokenized_dataset = dataset.map(tokenizeInputs)


# **Shuffle Dataset:**

# In[10]:


tokenized_dataset = tokenized_dataset.shuffle()


# **Count of Tokens in the Training Set:**

# In[11]:


token_count = 0
for sample in tokenized_dataset["train"]:
    token_count = token_count + len(sample["labels"])
    
print("Tokens in Training Set:", token_count)


# **Remove unnecessary columns:**

# In[12]:


#tokenized_dataset = tokenized_dataset.remove_columns(["tokens", "ner_tags", "langs", "spans"])


# **Save processed Dataset:**

# In[14]:


data_path = "./data/dataset_processed.pkl"
with open(data_path, 'wb') as pickle_file:
    pickle.dump(obj = tokenized_dataset, file=pickle_file)


# In[ ]:




