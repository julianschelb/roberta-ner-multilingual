#!/usr/bin/env python
# coding: utf-8

# # Finetuning RoBERTa for Token Classification: Testing

# ***

# ## Imports

# In[4]:


from transformers import (BertTokenizerFast,
                          RobertaTokenizerFast,
                          AutoTokenizer,
                          BertForTokenClassification,
                          RobertaForTokenClassification,
                          DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, 
                          TrainingArguments, Trainer)
from datasets import load_dataset, concatenate_datasets
import torch
import os


# ## Use Pretrained Model

# **Load Model and Tokenizer:**
# 
# Information about model variants can be found here: https://huggingface.co/docs/transformers/model_doc/roberta

# In[5]:


model_name = "xlm-roberta-large" #"bert-base-multilingual-cased" #xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", add_prefix_space=True) #AutoTokenizer(use_fast = True)
model = AutoModelForTokenClassification.from_pretrained(f"{model_name}")


# In[6]:


tokenizer.is_fast


# In[7]:


model.config


# **Predict Token Classification:**
# 
# Since the model has not been fintuned for Token classification yet, the prediction is poor as expected.

# In[8]:


text = """
Für Richard Phillips Feynman war es immer wichtig, die unanschaulichen 
Gesetzmäßigkeiten der Quantenphysik Laien und Studenten nahezubringen und verständlich zu machen.
"""


# In[9]:


inputs = tokenizer(
    text, add_special_tokens=False, return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
print(predicted_tokens_classes)


# In[ ]:




