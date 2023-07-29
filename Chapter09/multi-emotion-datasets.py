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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, jaccard_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments
)
import torch
from google.colab import drive

drive.mount("/content/gdrive/", force_remount=True)

BASE_PATH = "PATH_TO_DATA_ON_GOOGLE_DRIVE"

def get_kwt_tweets_dataset(code):
    if code == "KWTM":
        training_file = "train-KWT-M.csv"
        test_file = "test-KWT-M.csv"
    else:
        training_file = "train-KWT-U.csv"
        test_file = "test-KWT-U.csv"

    # add the base path
    training_file = f"{BASE_PATH}/{training_file}"
    test_file = f"{BASE_PATH}/{test_file}"

    data_files = {"train": training_file, "validation": test_file}
    ds = datasets.load_dataset("csv", data_files=data_files, delimiter=",", encoding='utf-8')

    dd = datasets.DatasetDict({"train":ds["train"], "validation":ds["validation"]})

    return dd

class Dataset(Enum):
    SEM11_AR=1
    SEM11_EN=2
    SEM11_ES=3
    KWT_M_AR=4
    KWT_U_AR=5

ds = Dataset.SEM11_EN

if (ds == Dataset.SEM11_AR):
    dataset = load_dataset("sem_eval_2018_task_1", "subtask5.arabic")
    model_name = "asafaya/bert-base-arabic"
elif (ds == Dataset.SEM11_EN):
    dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
    model_name = "bert-base-cased"
elif(ds == Dataset.SEM11_ES):
    dataset = load_dataset("sem_eval_2018_task_1", "subtask5.spanish")
    model_name = "dccuchile/bert-base-spanish-wwm-cased"
elif(ds == Dataset.KWT_M_AR):
    dataset = get_kwt_tweets_dataset("KWTM")
    model_name = "asafaya/bert-base-arabic"
elif(ds == Dataset.KWT_U_AR):
    dataset = get_kwt_tweets_dataset("KWTU")
    model_name = "asafaya/bert-base-arabic"

labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(labels)
print(id2label)
print(label2id)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenise_function(tweets):
    text = tweets["Tweet"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    labels_batch = {k: tweets[k] for k in tweets.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()

    return encoding

encoded_dataset = dataset.map(tokenise_function,
                              batched=True,
                              remove_columns=
                              dataset['train'].column_names
                              )

encoded_dataset.set_format("torch")

# the model
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'macro')
    recall = recall_score(y_true, y_pred, average = 'macro')
    jaccard = jaccard_score(y_true, y_pred, average='macro')

    # return as dictionary
    metrics = {
                'precision': precision,
                'recall': recall,
                'f1_micro_average': f1_micro_average,
                'f1_macro_average': f1_macro_average,
                'jaccard': jaccard,
                'accuracy': accuracy
              }

    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

metric_name = "jaccard"

training_args = TrainingArguments(
    model_name,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

# evaluation
trainer.evaluate()
