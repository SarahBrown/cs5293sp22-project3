import csv
import re
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
stopwords = nlp.Defaults.stop_words

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def process_sent(df_part):
    y = df_part.drop(['context'],axis=1)

    df_part = clean_context(df_part, 'context', 'context_tidy')
    df_part['redact_len'] = df_part['context'].apply(lambda x: count_redact(x))
    # df_part['left'] = df_part['context'].apply(lambda x: count_redact(x))
    # df_part['right'] = df_part['context'].apply(lambda x: count_redact(x))


    X = df_part.drop(['name', 'context'],axis=1)
    return X, y

def clean_context(df, col, tidy_col):
    #df[tidy_col] = df[col].apply(lambda x: x.replace('\u2588','')) # removes redacted text
    df[tidy_col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) # removes stop words
    df[tidy_col] = df[tidy_col].apply(lambda x: re.sub('[^a-zA-Z\'\"]', ' ', x)) # removes punctuation
    df[tidy_col] = df[tidy_col].apply(lambda x: re.sub('[\'\"]', '', x)) # removes punctuation
    df[tidy_col] = df[tidy_col].apply(lambda x: re.sub(' +', ' ', x)) # removes extra spaces

    return df

def count_redact(str):
    count = 0
    for i in str:
        if (i == '\u2588'):
            count += 1

    return count

def load_data(local):
    data = []

    if (local): 
        # loads data from tsv and separates based on data type (train, test, valid)
        with open('resources/unredactor.tsv') as file:
            tsv_file = csv.reader(file, delimiter="\t")
            
            # printing data line by line
            for line in tsv_file:
                data.append(line)

    # turns lists of lists into pandas df
    data_df = pd.DataFrame(data, columns = ['github', 'type', 'name', 'context'])
    data_df = data_df.dropna()

    return data_df


def train_model(X, y):
    print("training")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X,y)

    print("trained")
    with open('model.pkl','wb') as f:
        pickle.dump(clf,f)

    print(clf.score(X,y))


def load_model():
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    return clf

def main():
    # load and convert data
    use_local_file = True
    data_df = load_data(use_local_file)
    X, y = process_sent(data_df)
    X = X.drop(['github'], axis=1)
    y = y.drop(['github'], axis=1)
    
    v = DictVectorizer(sparse=False)
    X_train = (X.loc[X['type'] == 'training']).drop(['type'], axis=1)
    X_test  = (X.loc[X['type'] == 'training']).drop(['type'], axis=1)
    X_valid = (X.loc[X['type'] == 'training']).drop(['type'], axis=1)

    y_train = list((y.loc[X['type'] == 'training']).drop(['type'], axis=1).to_numpy().flatten())
    y_test  = list((y.loc[X['type'] == 'training']).drop(['type'], axis=1).to_numpy().flatten())
    y_valid = list((y.loc[X['type'] == 'training']).drop(['type'], axis=1).to_numpy().flatten())

    X_train = v.fit_transform(X_train.to_dict('records'))
    X_test  = v.transform( X_test.to_dict('records'))
    X_valid = v.transform(X_valid.to_dict('records'))

    # train_model(X_train, y_train)
    clf = load_model()
    print(clf.score(X_valid, y_valid))
    print(clf.score(X_test, y_test))
    
if __name__ == "__main__":
    main()