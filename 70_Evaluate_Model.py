# %% [markdown]
# # Finetuning RoBERTa for NER: Evaluate Model
#  

# %% [markdown]
# ***

# %% [markdown]
# ## Imports

# %%
from transformers import (BertTokenizerFast,
                          RobertaTokenizerFast,
                          AutoTokenizer,
                          BertForTokenClassification,
                          RobertaForTokenClassification,
                          DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, 
                          TrainingArguments, Trainer)
from datasets import load_dataset, load_metric, concatenate_datasets, DatasetDict
from pprint import pprint
import numpy as np
import pickle
import torch
import os

# %% [markdown]
# ## Load Dataset

# %%
data_path = "./data/dataset_processed.pkl"
with open(data_path, 'rb') as pickle_file:
    dataset = pickle.load(file=pickle_file)

# %% [markdown]
# ## Load Model and Tokenizer

# %% [markdown]
# Information about model variants can be found here: https://huggingface.co/docs/transformers/model_doc/roberta
# 
# Load Model which was finetuned:

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
label_list = dataset["train"].features[f"ner_tags"].feature.names

# %%
# model_name = "xlm-roberta-large" #"bert-base-multilingual-cased" #xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-final/", add_prefix_space=True) #AutoTokenizer(use_fast = True)
model = AutoModelForTokenClassification.from_pretrained("./results/checkpoint-final/")

# %% [markdown]
# **Define Metrics:**
# 
# See https://huggingface.co/course/chapter7/2#metrics

# %%
metric = load_metric("seqeval")

# %%
print(dataset["train"][150])

# %%
example = dataset["train"][150]
labels = [label_list[i] for i in example[f"ner_tags"]]
metric.compute(predictions=[labels], references=[labels])

# %% [markdown]
# **Calculate Accuracy:**

# %%
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# %%
predictions, labels, _ = trainer.predict(dataset["test"])
predictions = np.argmax(predictions, axis=-1)

# %%
label_names = dataset["train"].features[f"ner_tags"].feature.names

# %%
true_labels = [
    [label_names[l] for l in label  if l != -100] 
    for label in labels
]

true_predictions = [
    [label_names[p] for (p, l) in zip(prediction, label)  if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
pprint(results)

# %%



