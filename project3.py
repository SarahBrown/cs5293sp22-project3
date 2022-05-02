import pandas as pd
import csv
import re

import sklearn
#import nltk
#from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
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
    df_part.to_csv('resources/test.tsv', sep="\t", index=False)

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

    X_dict = (X.T.to_dict().values())
    #print(X_dict)
    vect = DictVectorizer(sparse=True)
    formatted_X = vect.fit_transform(X_dict)
    #print(vect.get_feature_names_out())

    X_out = X_data.drop(['github', 'type'],axis=1)
    X_out = X_out.to_numpy().flatten()
    
    arr = X.to_numpy().flatten()
    formatted_X = arr


    # vectorizer = CountVectorizer()
    # corpus = arr
    # tokenizer = vectorizer.build_tokenizer()
    # vocab = [tokenizer(str) for str in corpus]
    # vectorizer.fit(corpus)

    # formatted_vec_X = vectorizer.transform(corpus)

    arr = X_len.to_numpy()
    formatted_X_len = arr

    y_value = y.drop(['github', 'type'],axis=1)
    arr = y_value.to_numpy().flatten()
    formatted_y = list(arr)

    return formatted_X, formatted_X_len, formatted_y

def train_model(X, X_len, y):
    print("training")
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    X_vec = vectorizer.transform(X)
    X_vec_arr = X_vec.toarray()
    arr = np.concatenate((X_vec_arr,X_len),axis=1)
    #clf = MLPClassifier(random_state=1, max_iter=500).fit(X_vec, y)
    #clf = LogisticRegression(solver='liblinear').fit(arr, y)
    clf = make_pipeline(TfidfVectorizer(max_features = 50000,smooth_idf=True), MLPClassifier(random_state=1, max_iter=500))
    #clf = make_pipeline(CountVectorizer(), MLPClassifier(random_state=1, max_iter=500))
    #clf.fit(X_vec,y)

    print("trained")
    with open('model.pkl','wb') as f:
        pickle.dump(clf,f)

    print(clf.score(X_vec,y))


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

    X_train = (X.loc[X['type'] == 'training'])
    y_train = (y.loc[y['type'] == 'training'])

    X_train, X_train_len, y_train = convert_to_dict_and_len(X_train, y_train)

    X_valid = (X.loc[X['type'] == 'validation'])
    y_valid = (y.loc[y['type'] == 'validation'])
    X_valid, X_valid_len, y_valid = convert_to_dict_and_len(X_valid, y_valid)

    X_test = (X.loc[X['type'] == 'testing'])
    y_test = (y.loc[y['type'] == 'testing'])
    X_test, X_test_len, y_test = convert_to_dict_and_len(X_test, y_test)

    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)

    X_vec_valid = vectorizer.transform(X_valid)
    X_arr_valid = X_vec_valid.toarray()
    X_valid_arr = np.concatenate((X_arr_valid,X_valid_len),axis=1)


    X_vec_test = vectorizer.transform(X_test)
    X_arr_test = X_vec_test.toarray()
    X_test_arr = np.concatenate((X_arr_test,X_test_len),axis=1)


    # # trains and saves model
    # train_model(X_train, X_train_len, y_train)

    # # loads model
    clf = load_model()
    print(clf.score(X_vec_valid,y_valid))
    print(clf.score(X_vec_test,y_test))
    predicts = clf.predict(X_vec_valid)
    for i in range(len(predicts)):
        if ((X_valid[i] == y_valid[i])): #(y_valid[i] in y_train) and 
            print("Valid:",i)
            print(f'{X_valid[i]}\n{y_valid[i]}\n{predicts[i]}')
            #print(f'Was X_valid[i] in test? {X_valid[i] in X_train}')


if __name__ == "__main__":
    main()