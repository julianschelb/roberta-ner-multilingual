# %% [markdown]
# # Finetuning RoBERTa for NER: Compile Corpus

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
from datasets import load_dataset, concatenate_datasets, DatasetDict
import pickle
import torch
import os

# %% [markdown]
# ## Load Tokenizer

# %% [markdown]
# **Load Model and Tokenizer:**
# 
# Information about model variants can be found here: https://huggingface.co/docs/transformers/model_doc/roberta

# %%
model_name = "xlm-roberta-large" #"bert-base-multilingual-cased" #xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", add_prefix_space=True) #AutoTokenizer(use_fast = True)
#model = AutoModelForTokenClassification.from_pretrained(f"{model_name}")

# %% [markdown]
# ## Download Dataset for Finetuning

# %% [markdown]
# See:
# * Dataset on Huggingface: https://huggingface.co/datasets/wikiann
# * Load Datasets: https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/loading_methods

# %%
# Specify list of languages
#languages = ["en","de", "fr", "es", "zh", "ne"]
languages = ["en", "de"]
#languages = ["en"]
#languages=["ace","af","als","am","an","ang","ar","arc","arz","as","ast","ay","az","ba","bar","bat-smg","be","be-x-old","bg","bh","bn","bo","br","bs","ca","cbk-zam","cdo","ce","ceb","ckb","co","crh","cs","csb","cv","cy","da","de","diq","dv","el","eml","en","eo","es","et","eu","ext","fa","fi","fiu-vro","fo","fr","frr","fur","fy","ga","gan","gd","gl","gn","gu","hak","he","hi","hr","hsb","hu","hy","ia","id","ig","ilo","io","is","it","ja","jbo","jv","ka","kk","km","kn","ko","ksh","ku","ky","la","lb","li","lij","lmo","ln","lt","lv","map-bms","mg","mhr","mi","min","mk","ml","mn","mr","ms","mt","mwl","my","mzn","nap","nds","ne","nl","nn","no","nov","oc","or","os","pa","pdc","pl","pms","pnb","ps","pt","qu","rm","ro","ru","rw","sa","sah","scn","sco","sd","sh","si","simple","sk","sl","so","sq","sr","su","sv","sw","szl","ta","te","tg","th","tk","tl","tr","tt","ug","uk","ur","uz","vec","vep","vi","vls","vo","wa","war","wuu","xmf","yi","yo","zea","zh","zh-classical","zh-min-nan","zh-yue"]

# %%
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

# %% [markdown]
# **Limit Dataset Size for Testing:**

# %%
# Sample a subset of datapoints
#num_samples = 1000
#sample_ids = list(range(0,num_samples))

# Reduce the size of the dataset
#dataset_train = dataset_train.select(sample_ids)
#dataset_valid = dataset_valid.select(sample_ids)
#dataset_test = dataset_test.select(sample_ids)

print("Training Examples:", len(dataset_train))

# %%
dataset = DatasetDict({
    "train":dataset_train,
    "test":dataset_test, 
    "validation":dataset_valid
    })

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



