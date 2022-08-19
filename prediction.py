import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from transformers import BertTokenizerFast, BertForSequenceClassification
import matplotlib as plt
import itertools

model_path = './fake-news-bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
max_length = 512


def get_prediction(text, convert_to_label=False):
    inputs = tokenizer(text, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

labels_df = pd.read_csv('./labels.csv')
test_df = pd.read_csv('./test.csv')
test_df_copy = test_df.copy()
y_true = labels_df['label']
test_df_copy["newtext"] = test_df_copy["author"].astype(str) + ":" + test_df_copy["title"].astype(str) + "-" + test_df_copy["text"].astype(str)
test_df_copy["label"] = test_df_copy["newtext"].apply(get_prediction)

pred_df = test_df_copy[["id", "label"]]
y_pred = pred_df["label"]
cm = confusion_matrix(y_pred, y_true)
print(f1_score(y_pred, y_true))
print(accuracy_score(y_pred, y_true))
plot_confusion_matrix(cm, classes=['Fake Data', 'Real Data'])