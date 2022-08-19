import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from collections import Counter


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


cols = ['id', 'title', 'author', 'text', 'label']
remove_col = ['id', 'author']
cat_feat = []
target_col = ['label']
text_final = ['title', 'text']

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
dict_words = Counter(stop_words)


def rmv_unused_col(dataframe, columns=remove_col):
    '''
    Removing the unused column in any dataframe
    Args:
    dataframe (pandas.df): the dataframe that you want to delete
    the unused columns
    '''
    dataframe = dataframe.drop(columns, axis=1)
    return dataframe


def rmv_null_cols(dataframe):
    '''
    Removing the rows with null features
    Args:
    dataframe(obj:pandas.df)
    '''
    for cols in text_final:
        dataframe.loc[dataframe[cols].isnull(), cols] = "None"

    return dataframe


def data_cleaning(dataframe):
    '''
    Cleaning the data and get it ready for training
    Args:
    dataframe(obj: pandas.df)
    '''
    df = rmv_unused_col(dataframe)
    df = rmv_null_cols(dataframe)
    return df


def text_cleaning(txt):
    '''
    Cleaning the text to be ready for lemmatization
    Args:
    txt(obj: str)
    '''
    # Removing all urls
    txt = str(txt).replace(r'http[\w:/\.]+', ' ')
    # Preserving characters and punc
    txt = str(txt).replace(r'[^\.\w\s]', ' ')
    # Removing any non-alphabatical char
    txt = str(txt).replace('[^a-zA-Z]', ' ')
    # Removing one or more whitespace char
    txt = str(txt).replace(r'\s\s+', ' ')
    txt = txt.lower().strip()
    return txt


def lemma_preprocess(txt):
    '''
    Lemmatizing is used instead of stemming
    since lemmatizing not only do word reduction
    but also, evaluates the language's lexicon to
    apply morphological analysis to words. This
    will reduce the words to the base or dictionary form of the word.
    Args:
    txt(obj: str)
    '''
    txt = text_cleaning(txt)
    wordlist = re.sub(r'[^\w\s]', '', txt).split()
    text = ' '.join([lemmatizer.lemmatize(word) for word
                     in wordlist if
                     word not in dict_words])
    return text
