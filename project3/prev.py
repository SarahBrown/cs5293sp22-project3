import pandas as pd
import csv
import re

import sklearn
#import nltk
#from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import numpy as np
import spacy
import en_core_web_lg
import pandas as pd

import pickle

nlp = en_core_web_lg.load()
stopwords = nlp.Defaults.stop_words

def process_sent(df_part):
    y = df_part.drop(['context'],axis=1)

    df_part = clean_context(df_part, 'context', 'context_tidy', True)

    X = df_part.drop(['name', 'context'],axis=1)
    return X, y

def clean_context(df, col, tidy_col, add_redact_len):
    #stop = stopwords.words('english') nltk
    #df[tidy_col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) # removes stop words
    df[tidy_col] = df[col].apply(lambda x: x.replace('\u2588','')) # removes redacted text
    df[tidy_col] = df[tidy_col].apply(lambda x: re.sub('[^a-zA-Z\'\"]', ' ', x)) # removes punctuation
    df[tidy_col] = df[tidy_col].apply(lambda x: re.sub('[\'\"]', '', x)) # removes punctuation
    df[tidy_col] = df[tidy_col].apply(lambda x: re.sub(' +', ' ', x)) # removes extra spaces

    if (add_redact_len):
        df['redact_len'] = df[col].apply(lambda x: count_redact(x))

    return df

def count_redact(str):
    count = 0
    for i in str:
        if (i == '\u2588'):
            count += 1

    return count

def load_data():
    data = []

    # loads data from tsv and separates based on data type (train, test, valid)
    with open('resources/data.tsv') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        
        # printing data line by line
        for line in tsv_file:
            data.append(line)

    # turns lists of lists into pandas df
    data_df = pd.DataFrame(data, columns = ['github', 'type', 'name', 'context'])

    return data_df

def convert_to_dict_and_len(X_data, y):
    X = X_data.drop(['github', 'type', 'redact_len'],axis=1)
    X_len = X_data.drop(['github', 'type', 'context_tidy'],axis=1)

    X_out = X_data.drop(['github', 'type'],axis=1)
    X_out = X_out.to_numpy().flatten()
    
    arr = X.to_numpy().flatten()
    formatted_X = arr

    arr = X_len.to_numpy()
    formatted_X_len = arr

    y_value = y.drop(['github', 'type'],axis=1)
    arr = y_value.to_numpy().flatten()
    formatted_y = list(arr)

    return formatted_X, formatted_X_len, formatted_y

def train_model(X , y):
    print("training")
    #clf = MLPClassifier(random_state=1, max_iter=500).fit(X_vec, y)
    #clf = LogisticRegression(solver='liblinear').fit(arr, y)
    #clf = make_pipeline(CountVectorizer(), MLPClassifier(random_state=1, max_iter=500))
    # clf = make_pipeline(CountVectorizer(), MLPClassifier(random_state=1, max_iter=500))
    
    clf = MLPClassifier(random_state=1, max_iter=50)
    clf.fit(X,y)

    print("trained")
    with open('model.pkl','wb') as f:
        pickle.dump(clf,f)

    print(clf.score(X,y))


def load_model():
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    return clf

def concat_len(arr, len_arr):
    concat_arr = np.concatenate((arr,len_arr),axis=1)
    return concat_arr

def main():
    # load and convert data
    data_df = load_data()
    X, y = process_sent(data_df)

    # X_train = (X.loc[X['type'] == 'training'])
    # y_train = (y.loc[y['type'] == 'training'])

    # X_train, X_train_len, y_train = convert_to_dict_and_len(X_train, y_train)

    # X_valid = (X.loc[X['type'] == 'validation'])
    # y_valid = (y.loc[y['type'] == 'validation'])
    # X_valid, X_valid_len, y_valid = convert_to_dict_and_len(X_valid, y_valid)

    # X_test = (X.loc[X['type'] == 'testing'])
    # y_test = (y.loc[y['type'] == 'testing'])
    # X_test, X_test_len, y_test = convert_to_dict_and_len(X_test, y_test)

    # vectorizer = CountVectorizer()
    # vectorizer.fit(X_train)

    # X_vec_train = vectorizer.transform(X_train)
    # X_train_arr = concat_len(X_vec_train.toarray(), X_train_len)
    
    # X_vec_valid = vectorizer.transform(X_valid)
    # X_valid_arr = concat_len(X_vec_valid.toarray(), X_valid_len)

    # X_vec_test = vectorizer.transform(X_test)
    # X_test_arr = concat_len(X_vec_test.toarray(), X_test_len)

    # # trains and saves model
    # train_model(X_train_arr, y_train)

    # # loads model
    # clf = load_model()

    # print(clf.score(X_valid_arr,y_valid))
    # print(clf.score(X_test_arr,y_test))

    # predicts = clf.predict(X_vec_valid)
    # for i in range(len(predicts)):
    #     if ((X_valid[i] == y_valid[i])): #(y_valid[i] in y_train) and 
    #         print("Valid:",i)
    #         print(f'{X_valid[i]}\n{y_valid[i]}\n{predicts[i]}')
    #         #print(f'Was X_valid[i] in test? {X_valid[i] in X_train}')


    df_selected = X.drop(['github', 'type', 'redact_len'], axis=1)
    df_features = df_selected.to_dict(orient='records')
    print(df_features)
    vectorizer = DictVectorizer()
    features = vectorizer.fit_transform(df_features).toarray()

    labels = y.drop(['github', 'type'],axis=1).to_numpy().flatten()

    # features_train, features_test, labels_train, labels_test = train_test_split(
    #     features, labels, 
    #     test_size=0.20, random_state=42)

    # train_model(features_train, labels_train)

    # clf = load_model()
    # print(clf.score(features_test,labels_test))
    # predicts = clf.predict(features_test)

    # labels = list(labels_train.flatten())
    # print("Correct")
    # for i in range(len(predicts)):
    #     if (labels_test[i] == predicts[i]):
    #         count = labels.count(labels_test[i])
    #         print(f'Expected: {labels_test[i]}. Actual: {predicts[i]}. Count in train: {count}')
    # print("Incorrect")
    # for i in range(len(predicts)):
    #     if (labels_test[i] != predicts[i]):
    #         count = labels.count(labels_test[i])
    #         print(f'Label: {labels_test[i]}. Model: {predicts[i]}. Count in train: {count}')

    # TODO Unredacted texts should go in the output folder location.

if __name__ == "__main__":
    main()