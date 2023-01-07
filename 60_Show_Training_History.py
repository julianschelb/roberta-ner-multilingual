# %% [markdown]
# # Finetuning RoBERTa for NER: Training History
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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import torch
import os

# %% [markdown]
# ## Load Training History:

# %%
data_path = "./results/checkpoint-final/training_args.pkl"
with open(data_path, 'rb') as pickle_file:
    training_args = pickle.load(file=pickle_file)

# %%
data_path = "./results/checkpoint-final/training_history.pkl"
with open(data_path, 'rb') as pickle_file:
    training_state = pickle.load(file=pickle_file)

# %% [markdown]
# ## Show Training Parameters

# %%
print(training_args)

# %% [markdown]
# ## Show Training History

# %%
print(training_state)

# %%
training_history = training_state.log_history

# Training
epochs = [epoch.get("epoch") for epoch in  training_history if epoch.get("loss") is not None]
steps = [epoch.get("step") for epoch in  training_history if epoch.get("loss") is not None]
loss = [epoch.get("loss") for epoch in  training_history if epoch.get("loss") is not None]

# Eval
eval_epochs = [epoch.get("epoch") for epoch in  training_history if epoch.get("eval_loss") is not None]
eval_steps = [epoch.get("step") for epoch in  training_history if epoch.get("eval_loss") is not None]
eval_loss = [epoch.get("eval_loss") for epoch in  training_history if epoch.get("eval_loss") is not None]
eval_precision = [epoch.get("eval_precision") for epoch in  training_history if epoch.get("eval_precision") is not None]
eval_recall = [epoch.get("eval_recall") for epoch in  training_history if epoch.get("eval_recall") is not None]
eval_f1 = [epoch.get("eval_recall") for epoch in  training_history if epoch.get("eval_f1") is not None]
eval_accuracy = [epoch.get("eval_accuracy") for epoch in  training_history if epoch.get("eval_accuracy") is not None]

# %%
try:
    p = sns.lineplot( x=steps, y=loss)
    p.set_xlabel("Training Steps")
    p.set_ylabel("Loss")
    plt.savefig('./figures/history_loss.png')
except Exception as e:
    print(e)

# %%
try: 
    p = sns.lineplot(x=eval_steps, y=eval_loss)
    p.set_xlabel("Eval Steps")
    p.set_ylabel("Loss")
    plt.savefig('./figures/history_eval_loss.png')
except Exception as e:
    print(e)

# %%
try: 
    p = sns.lineplot(x=eval_steps, y=eval_accuracy)
    p.set_xlabel("Eval Steps")
    p.set_ylabel("Accuracy")
    plt.savefig('./figures/history_eval_accuracy.png')
except Exception as e:
    print(e)

# %%



