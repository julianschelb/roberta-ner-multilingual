#!/usr/bin/env python
# coding: utf-8

# # Finetuning RoBERTa for NER: Train Model
#  

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
from datasets import load_dataset, load_metric, concatenate_datasets, DatasetDict
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import os


# ## Load Dataset

# In[2]:


data_path = "./data/dataset_processed.pkl"
with open(data_path, 'rb') as pickle_file:
    dataset = pickle.load(file=pickle_file)


# In[3]:


dataset["train"][0]


# ## Load Model and Tokenizer

# Information about model variants can be found here: https://huggingface.co/docs/transformers/model_doc/roberta
# 
# Load Model which can be finetuned:

# In[4]:


import gc
gc.collect()
torch.cuda.empty_cache()


# In[5]:


label_list = dataset["train"].features[f"ner_tags"].feature.names


# In[6]:


model_name = "xlm-roberta-large" #"bert-base-multilingual-cased" #xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", add_prefix_space=True) #AutoTokenizer(use_fast = True)
model = AutoModelForTokenClassification.from_pretrained(f"{model_name}", num_labels=len(label_list))


# ## Define Data Collator

# In[7]:


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# ## Define Trainer

# About the Model:
# 
# see https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/modeling_utils.py#L829

# In[8]:


print("Parameters:", model.num_parameters())
print("Expected Input Dict:", model.main_input_name )

# Estimate FLOPS needed for one training example
sample = dataset["train"][0]
sample["input_ids"] = torch.Tensor(sample["input_ids"])
flops_est = model.floating_point_ops(input_dict = sample, exclude_embeddings = False)

print("FLOPS needed per Training Sample:", flops_est )


# In[9]:


dataset


# **Define Optimizer:**
# 
# See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor

# In[10]:


#from transformers.optimization import Adafactor, AdafactorSchedule

#optimizer = Adafactor(    
#        model.parameters(),
#        lr=1e-3,
#        eps=(1e-30, 1e-3),
#        clip_threshold=1.0,
#        decay_rate=-0.8,
#        beta1=None,
#        weight_decay=0.0,
#        relative_step=False,
#        scale_parameter=False,
#        warmup_init=False,
#    )

#lr_scheduler = AdafactorSchedule(optimizer)


# **Define Metrics:**
# 
# See https://huggingface.co/course/chapter7/2#metrics

# In[11]:


metric = load_metric("seqeval")


# In[12]:


example = dataset["train"][150]
labels = [label_list[i] for i in example[f"labels"]]
metric.compute(predictions=[labels], references=[labels])


# Set correct class labels:

# In[13]:


label_names = dataset["train"].features[f"ner_tags"].feature.names

id2label = {id : label for id, label in enumerate(label_names)}
label2id = {label: id for id, label in enumerate(label_names)}

model.config.id2label = id2label
model.config.label2id = label2id


# Define callback function to evaluate the model:

# In[14]:


label_names = model.config.id2label

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_names[l] for l in label  if l != -100] for label in labels]
    #true_predictions = [model.config.id2label[t.item()] for t in predictions]
    
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label)  if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# **Remove unnecessary columns:**

# In[15]:


dataset = dataset.remove_columns(["tokens", "ner_tags", "langs", "spans"])


# **Set further Training Arguments:**
# 
# See https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/trainer#transformers.TrainingArguments

# In[16]:


training_args = TrainingArguments(
    output_dir="./results",
    save_strategy= "no",# "epoch",
    #save_steps = 2000,
    remove_unused_columns = True,
    evaluation_strategy="steps",
    eval_steps = 2000,
    #load_best_model_at_end=True,
    logging_strategy = "steps",
    logging_steps = 2000,
    learning_rate=2e-5,
    #auto_find_batch_size = True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    #optim="adamw_torch",
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
    #fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    #optimizers=(optimizer, lr_scheduler),
    compute_metrics=compute_metrics
)


# ## Train Model
# 
# GPU used by Kaggle: https://www.nvidia.com/de-de/data-center/tesla-p100/

# In[17]:


get_ipython().system('nvidia-smi')


# In[18]:


trainer.train()


# In[ ]:


eval_results = trainer.evaluate()
print(f"Eval Loss: {eval_results['eval_loss']}")


# **Saving the fine tuned model & tokenizer:**

# In[ ]:


trainer.save_model(f'./results/checkpoint-final/')


# **Calculate Accuracy:**

# In[ ]:


predictions, labels, _ = trainer.predict(dataset["test"])
predictions = np.argmax(predictions, axis=-1)


# In[ ]:


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


# In[ ]:


training_history = trainer.state.log_history

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

eval_accuracy


# In[ ]:


import seaborn as sns


# In[ ]:


p = sns.lineplot( x=steps, y=loss)
p.set_xlabel("Training Steps")
p.set_ylabel("Loss")
plt.savefig('history_loss.png')


# In[ ]:


p = sns.lineplot(x=eval_steps, y=eval_loss)
p.set_xlabel("Eval Steps")
p.set_ylabel("Loss")
plt.savefig('history_eval_loss.png')


# In[ ]:


p = sns.lineplot(x=eval_steps, y=eval_accuracy)
p.set_xlabel("Eval Steps")
p.set_ylabel("Accuracy")
plt.savefig('history_eval_accuracy.png')

