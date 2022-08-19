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

<img style="float: left" src="/index.png">

The confusion matrix can bring out the recall and precision of the model and shows how the overall performance of the model is in detecting the classes.

## Future Improvements
In order to improve the model performance the number of epochs can go higher and the batch size can be changed. Also, more models should be tested
in order to find a better model.

## Requirements
To install the requirements enter the following code after cloning the project:\
```pip install -r requirements.txt```\
Make sure the data is in the same repo as the cloned repo and it is unzipped to run the program.\
to run preprocessing and train the model you can run\
```
python3 train.py```\
To get the prediction for the test and all the evaluation run:\
```
python3 prediction.py```\
it is strongly advised to make a virtual environment to install the dependencies.
