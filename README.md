# fake-news-detection

## The overall approach
The overall approach to solve this problem was to lemmatize the text and the title in order to be used with Bert Sequence Classification
which I did a binary classification. The distributation of the fake to reliable news was checked which showed the dataset is balanced.

## Preprocessing approach
In order to make the data ready for trainig the null elements were deleted, it was detected that the id and author of the dataset was removed
as they aren't significant predictors. The text was cleaned from any non-alphabetical characters, whitespaces, urls. At the end the texts and title
of the news was lemmatized in order to make it ready for training. The text and title in datasets were split into training and validation sets to
train and evaluate the model. The model was then saved as 'fake-news-bert-base-uncased' and loaded in order to be tested on the test dataset.

## Evaluation procedure and metrics
The model was tested based on three metrics:
**F1 Score**: The f1 score showed 65%.
**Accuracy Score**: The accuracy score was 63%
**Confusion Matrix**: The confusion matrix was drawn and the result is as followed:
![confusion-matrix](index.jpg)