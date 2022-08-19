import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from preprocess import lemma_preprocess, data_cleaning
import random


class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def set_seed(seed):
    '''
    Helper function for reproducible behavior to set the seed in
    random, numpy, torch, and/or tf (if installed)
    '''
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        # It is safe to call this function even if cuda is not available
        torch.cuda.manual_seed_all(seed)
    if is_tf_available():
        tf.random.set_seed(seed)


def prepare_data(df, test_size=0.2, include_title=True, include_author=True):
    texts = []
    labels = []
    for i in range(len(df)):
        text = df["text"].iloc[i]
        label = df["label"].iloc[i]
        if include_title:
            text = df["title"].iloc[i] + " - " + text
        if include_author:
            text = df["author"].iloc[i] + " : " + text
        if text and label in [0, 1]:
            texts.append(text)
            labels.append(label)
    return train_test_split(texts, labels, test_size=test_size)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(1)
df = pd.read_csv('./train.csv')
news_df = data_cleaning(df)
news_df['text'] = news_df.text.apply(lemma_preprocess)
news_df['title'] = news_df.title.apply(lemma_preprocess)
news_df = news_df[news_df['text'].notna()]
news_df = news_df[news_df['author'].notna()]
news_df = news_df[news_df['title'].notna()]

model_name = "bert-base-uncased"
max_length = 512
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)


train_texts, valid_texts, train_labels, valid_labels = prepare_data(news_df)
# Encoding the training texts by tokenizing them
train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                            max_length=max_length)
# Encoding the validation texts by tokenizing them
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True,
                            max_length=max_length)
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=10,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,               # log & save weights each logging_steps
    save_steps=200,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train()
trainer.evaluate()
model_path = 'fake-news-bert-base-uncased'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

