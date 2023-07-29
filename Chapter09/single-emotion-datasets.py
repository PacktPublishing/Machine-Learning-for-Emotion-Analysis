# -*- coding: utf-8 -*-
!pip install datasets
!pip install evaluate
!pip install transformers

import datasets
from datasets import load_dataset
from enum import Enum
import evaluate
from evaluate import evaluator
import numpy as np
from sklearn import metrics
from sklearn.metrics import jaccard_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    Trainer,
    TrainingArguments
)
import pandas as pd
from pathlib import Path
from google.colab import drive

drive.mount("/content/gdrive/", force_remount=True)

BASE_PATH = "PATH_TO_DATA_ON_GOOGLE_DRIVE"

class Dataset(Enum):
    SEM4_EN=1
    WASSA_EN=2
    CARER_EN=3
    SEM4_AR=4
    SEM4_ES=5
    aclImdb_EN=6

# set the required dataset here
ds = Dataset.SEM4_EN

NUM_LABELS = 4
COLS = 'ID', 'tweet', 'label'

if (ds == Dataset.SEM4_EN):
    training_file = "SEM4_EN_train.csv"
    test_file = "SEM4_EN_dev.csv"
    training_file = "t_SEM4_EN_train_one_column.csv"
    test_file = "t_SEM4_EN_dev_one_column.csv"
elif (ds == Dataset.WASSA_EN):
    training_file = "WASSA_train.csv"
    test_file = "WASSA_dev.csv"
elif(ds == Dataset.CARER_EN):
    training_file = "CARER_EN_train.csv"
    test_file = "CARER_EN_dev.csv"
    NUM_LABELS = 6
elif(ds == Dataset.SEM4_ES):
    training_file = "SEM4_ES_train.csv"
    test_file = "SEM4_ES_dev.csv"
    NUM_LABELS = 5
elif(ds == Dataset.SEM4_AR):
    training_file = "SEM4_AR_train.csv"
    test_file = "SEM4_AR_dev.csv"
elif(ds == Dataset.aclImdb_EN):
    NUM_LABELS = 2
    training_file = "aclImdb_EN_train.csv"
    test_file = "aclImdb_EN_dev.csv"

# select a model
if "_AR_" in training_file:
    model_name = "asafaya/bert-base-arabic"
elif "_EN_" in training_file:
     model_name = "bert-base-cased"
elif "_ES_" in training_file:
     model_name = "dccuchile/bert-base-spanish-wwm-cased"

# add the base path
training_file = f"{BASE_PATH}/{training_file}"
test_file = f"{BASE_PATH}/{test_file}"

# get file name for saving
stub = (Path(training_file).stem)

def get_tweets_dataset():
    data_files = {"train": training_file, "test": test_file}
    ds = datasets.load_dataset("csv", data_files=data_files, delimiter=",", encoding='utf-8')

    ds_columns = ds['train'].column_names
    drop_columns = [x for x in ds_columns if x not in COLS]
    ds = ds.remove_columns(drop_columns)

    dd = datasets.DatasetDict({"train":ds["train"], "test":ds["test"]})

    return dd

dataset = get_tweets_dataset()

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenise_function(tweets):
    return tokenizer(tweets["tweet"], padding="max_length", truncation=True, max_length = 512)

tokenised_datasets = dataset.map(tokenise_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
training_args = TrainingArguments(output_dir=f"{stub}")

training_args = TrainingArguments(output_dir=f"{stub}", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_datasets["train"],
    eval_dataset=tokenised_datasets["test"],
)

trainer.train()

trainer.save_model(stub)

predictions = trainer.predict(tokenized_datasets["test"])

def get_jaccard_score(predictions, references, average="macro"):
    return {
        "jaccard": float(
            jaccard_score(references, predictions, average=average)
        )
    }

model_predictions = np.argmax(predictions.predictions, axis=1)
model_predictions = model_predictions.tolist()

model_references = tokenised_datasets["test"]["label"]

measures = [
              ["precision" , "macro"],
              ["recall" , "macro"],
              ["f1" , "micro"],
              ["f1" , "macro"],
              ["jaccard" , "macro"],
              ["accuracy" , None],
            ]

for measure in measures:
    measure_name = measure[0]
    average = measure[1]
    if measure_name=="jaccard":
        results = get_jaccard_score(references=model_references, predictions=model_predictions, average=average)
    else:
        metric = evaluate.load(measure_name)
        if measure_name=="accuracy":
            results = metric.compute(references=model_references, predictions=model_predictions)
        else:
            results = metric.compute(references=model_references, predictions=model_predictions, average=average)

print(measure_name, average, results[measure_name])
