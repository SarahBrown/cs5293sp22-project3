import csv
import re
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
stopwords = nlp.Defaults.stop_words

def process_sent(df_part):
    y = df_part.drop(['context','github'],axis=1)

    df_part = clean_context(df_part)
    df_part['no_stop'] =  df_part['context_tidy'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) # removes stop words
    X_nostop = df_part.drop(['name','context','context_tidy','github','redact_len'],axis=1)
    df_part['pos_window'] = df_part['context_tidy'].apply(lambda x: make_pos_window(x))    
    df_part['word-2'] = df_part['pos_window'].apply(lambda x: x[0])
    df_part['pos-2'] = df_part['pos_window'].apply(lambda x: x[1])
    df_part['word-1'] = df_part['pos_window'].apply(lambda x: x[2])
    df_part['pos-1'] = df_part['pos_window'].apply(lambda x: x[3])
    df_part['word+1'] = df_part['pos_window'].apply(lambda x: x[4])
    df_part['pos+1'] = df_part['pos_window'].apply(lambda x: x[5])
    df_part['word+2'] = df_part['pos_window'].apply(lambda x: x[6])
    df_part['pos+2'] = df_part['pos_window'].apply(lambda x: x[7])

    X = df_part.drop(['name', 'context','context_tidy','pos_window','no_stop','github'],axis=1)
    return X, y, X_nostop

def clean_context(df):
    df['redact_len'] = df['context'].apply(lambda x: count_redact(x))
    df['context_tidy'] = df['context'].apply(lambda x: re.sub(r'(\w+)n\'t', r'\g<1>' + " not", x))
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('\u2588+','REDACTED',x)) # removes redacted text
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('[^a-zA-Z \' \"]', ' ', x)) # removes punctuation
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('[\'\"]', '', x)) # removes punctuation
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub(' +', ' ', x)) # replaces extra spaces

    return df

def count_redact(str):
    count = 0
    for i in str:
        if (i == '\u2588'):
            count += 1

    return count

def make_pos_window(context):
    # context = re.sub(r'(\w+)n\'t', r'\g<1>' + " not", context)
    # context = re.sub(r'\u2588+', "REDACTED", context)
    # context = re.sub(r'[^a-zA-Z ]', "", context)

    doc = nlp(context)
    pos = ["NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL"]

    redact_index = [x for x in range(len(doc)) if doc[x].text == "REDACTED"] # gets list of elements matching REDACTED
    if (len(redact_index) > 0):
        redact_index = redact_index[0] # gets index of redacted string
    else:
        return pos

    if (redact_index - 2 > 0):
        pos[0] = doc[redact_index-2].text
        pos[1] = doc[redact_index-2].pos_

    if (redact_index - 1 > 0):
        pos[2] = doc[redact_index-1].text
        pos[3] = doc[redact_index-1].pos_


    if (redact_index + 1 < len(doc)):
        pos[4] = doc[redact_index+1].text
        pos[5] = doc[redact_index+1].pos_

    if (redact_index + 2 < len(doc)):
        pos[6] = doc[redact_index+2].text
        pos[7] = doc[redact_index+2].pos_

    return pos

def load_data(local):
    data = []

    if (local): 
        # loads data from tsv and separates based on data type (train, test, valid)
        with open('resources/unredactor.tsv') as file:
            tsv_file = csv.reader(file, delimiter='\t+')
            
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
    #clf = MLPClassifier(random_state=1, max_iter=500)
    clf.fit(X,y)

    print("trained")

    with open('model.pkl','wb') as f:
        pickle.dump(clf,f)
    print("model saved")

    print(clf.score(X,y))


def how_much_data_overlap(y_train, y_test, y_valid):
    y_train_set = set()
    for i in y_train:
        y_train_set.add(i)

    y_test_set = set()
    for i in y_test:
        y_test_set.add(i)

    y_valid_set = set()
    for i in y_valid:
        y_valid_set.add(i)

    train_total = 0
    test_total = 0
    valid_total = 0

    y_train_summary = []
    for i in y_train_set:
        count = y_train.count(i)
        train_total += count
        y_train_summary.append([i, count])

    y_test_summary = []
    for i in y_test_set:
        count = y_train.count(i)
        test_total += count
        y_test_summary.append([i, count])

    y_valid_summary = []
    for i in y_valid_set:
        count = y_valid.count(i)
        valid_total += count
        y_valid_summary.append([i, count])

    train_sorted = sorted(y_train_summary, key=lambda x: x[1])
    test_sorted = sorted(y_test_summary, key=lambda x: x[1])
    valid_sorted = sorted(y_valid_summary, key=lambda x: x[1])
    print('train')
    print(train_sorted)
    print('test')
    print(test_sorted)
    print('valid')
    print(valid_sorted)
    print(f'Test overlap with train: {test_total}. Valid overlap with train: {valid_total}.')

def load_model():
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    return clf

def main():
    # load and convert data
    use_local_file = True
    data_df = load_data(use_local_file)
    X, y, X_nostop = process_sent(data_df)
    
    dict = DictVectorizer(sparse=False)
    X_train = (X.loc[X['type'] == 'training']).drop(['type'], axis=1)
    X_test  = (X.loc[X['type'] == 'testing']).drop(['type'], axis=1)
    X_valid = (X.loc[X['type'] == 'validation']).drop(['type'], axis=1)

    # X_train_nostop = (X_nostop.loc[X_nostop['type'] == 'training']).drop(['type',], axis=1)
    # X_train_nostop = np.reshape(X_train_nostop, (X_train_nostop.shape[0],1))
    # X_test_nostop  = (X_nostop.loc[X_nostop['type'] == 'testing']).drop(['type',], axis=1)
    # X_valid_nostop = (X_nostop.loc[X_nostop['type'] == 'validation']).drop(['type',], axis=1)

    y_train = list((y.loc[X['type'] == 'training']).drop(['type'], axis=1).to_numpy().flatten())
    y_test  = list((y.loc[X['type'] == 'testing']).drop(['type'], axis=1).to_numpy().flatten())
    y_valid = list((y.loc[X['type'] == 'validation']).drop(['type'], axis=1).to_numpy().flatten())
    # how_much_data_overlap(y_train, y_test, y_valid)

    X_train = dict.fit_transform(X_train.to_dict('records'))
    X_test = dict.transform(X_test.to_dict('records'))
    X_valid = dict.transform(X_valid.to_dict('records'))

    # tfidf = TfidfVectorizer()
    # print(len(X_train_nostop.to_numpy().flatten()))
    # X_train_tfidf = tfidf.fit_transform(X_train_nostop.to_numpy().flatten())
    # X_train_tfidf = tfidf.transform(X_test_nostop.to_numpy().flatten())
    # X_train_tfidf = tfidf.transform(X_valid_nostop.to_numpy().flatten())

    # print(X_train_tfidf.shape)


    train_model(X_train, y_train)
    clf = load_model()
    print(clf.score(X_valid, y_valid))
    print(clf.score(X_test, y_test))

    # predicts = clf.predict(X_valid)
    # for i in range(len(predicts)):
    #     if (y_train.count(y_valid[i]) >= 1):
    #         if (y_valid[i] != predicts[i]):
    #             print("Valid:",i)
    #             print(f'Data says:  {y_valid[i]}\nModel says: {predicts[i]}') #{X_valid[i]}\n
    #             print(f'Was y_valid[i] in train? {y_valid[i] in y_train}. Count: {y_train.count(y_valid[i])}')
 
    # predicts = clf.predict(X_test)
    # for i in range(len(predicts)):
    #     if (y_train.count(y_test[i]) >= 1):
    #         if (y_test[i] != predicts[i]):
    #             print("Test:",i)
    #             print(f'Data says:  {y_test[i]}\nModel says: {predicts[i]}') #{X_valid[i]}\n
    #             print(f'Was y_test[i] in train? {y_test[i] in y_train}. Count: {y_train.count(y_test[i])}')
    
if __name__ == "__main__":
    main()