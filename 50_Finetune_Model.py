# %% [markdown]
# # Finetuning RoBERTa for NER: Train Model
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
import numpy as np
import dill as pickle
import torch
import math
import os

# %% [markdown]
# ## Load Dataset

# %%
data_path = "./data/dataset_processed.pkl"
with open(data_path, 'rb') as pickle_file:
    dataset = pickle.load(file=pickle_file)

# %%
dataset["train"][0]

# %% [markdown]
# ## Load Model and Tokenizer

# %% [markdown]
# Information about model variants can be found here: https://huggingface.co/docs/transformers/model_doc/roberta
# 
# Load Model which can be finetuned:

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
label_list = dataset["train"].features[f"ner_tags"].feature.names

# %%
model_name = "xlm-roberta-large" #"bert-base-multilingual-cased" #xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", add_prefix_space=True) #AutoTokenizer(use_fast = True)
model = AutoModelForTokenClassification.from_pretrained(f"{model_name}", num_labels=len(label_list))

# %% [markdown]
# ## Define Data Collator

# %%
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# %% [markdown]
# ## Define Trainer

# %% [markdown]
# About the Model:
# 
# see https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/modeling_utils.py#L829

# %%
print("Parameters:", model.num_parameters())
print("Expected Input Dict:", model.main_input_name )

# Estimate FLOPS needed for one training example
sample = dataset["train"][0]
sample["input_ids"] = torch.Tensor(sample["input_ids"])
flops_est = model.floating_point_ops(input_dict = sample, exclude_embeddings = False)

print("FLOPS needed per Training Sample:", flops_est )

# %%
dataset

# %% [markdown]
# **Define Optimizer:**
# 
# See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor

# %%
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

num_epochs = 1
batch_size = 16
num_reports = 10

# A training step is one gradient update. In one step batch_size examples are processed.
# An epoch consists of one full cycle through the training data. 
# This is usually many steps. As an example, if you have 2,000 images and use
# a batch size of 10 an epoch consists of:
gpu_count = torch.cuda.device_count()
num_steps = (len(dataset["train"]) / batch_size / gpu_count) * num_epochs

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-6, weight_decay=0.01, no_deprecation_warning= True)

scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps= num_steps 
)

print("Steps:", num_steps)

# %% [markdown]
# **Define Log and Eval Interval:**

# %%
report_steps = math.floor(num_steps / num_reports)
print("Eval interval:", report_steps)

# %% [markdown]
# **Define Metrics:**
# 
# See https://huggingface.co/course/chapter7/2#metrics

# %%
metric = load_metric("seqeval")

# %%
example = dataset["train"][150]
labels = [label_list[i] for i in example[f"labels"]]
metric.compute(predictions=[labels], references=[labels])

# %% [markdown]
# Set correct class labels:

# %%
label_names = dataset["train"].features[f"ner_tags"].feature.names

id2label = {id : label for id, label in enumerate(label_names)}
label2id = {label: id for id, label in enumerate(label_names)}

model.config.id2label = id2label
model.config.label2id = label2id

# %% [markdown]
# Define callback function to evaluate the model:

# %%
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

# %% [markdown]
# **Remove unnecessary columns:**

# %%
dataset = dataset.remove_columns(["tokens", "ner_tags", "langs", "spans"])

# %% [markdown]
# **Set further Training Arguments:**
# 
# See https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/trainer#transformers.TrainingArguments

# %%
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy = "steps",
    save_steps = report_steps,
    remove_unused_columns = True,
    evaluation_strategy="steps",
    eval_steps = report_steps,
    #load_best_model_at_end=True,
    logging_strategy = "steps",
    logging_steps = report_steps,
    #learning_rate= 2e-5,
    #auto_find_batch_size = True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #gradient_accumulation_steps=4,
    #optim="adamw_torch",
    num_train_epochs=num_epochs,
    #weight_decay=0.01,
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics
)

# %% [markdown]
# ## Train Model
# 
# GPU used by Kaggle: https://www.nvidia.com/de-de/data-center/tesla-p100/

# %%
# nvidia-smi

# %%
trainer.train()

# %%
eval_results = trainer.evaluate()
print(f"Eval Loss: {eval_results['eval_loss']}")

# %% [markdown]
# **Saving the fine tuned model & tokenizer:**

# %%
trainer.save_model(f'./results/checkpoint-final/')

# %% [markdown]
# **Save Training History:**

# %%
data_path = "./results/checkpoint-final/training_args.pkl"
with open(data_path, 'wb') as pickle_file:
    pickle.dump(obj = trainer.args, file=pickle_file)

# %%
data_path = "./results/checkpoint-final/training_history.pkl"
with open(data_path, 'wb') as pickle_file:
    pickle.dump(obj = trainer.state, file=pickle_file)

# %% [markdown]
# **Calculate Accuracy:**

# %%
predictions, labels, _ = trainer.predict(dataset["test"])
predictions = np.argmax(predictions, axis=-1)

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



