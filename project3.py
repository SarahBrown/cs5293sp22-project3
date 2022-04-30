import pandas as pd
import csv
import re

import sklearn
import nltk
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier

def process_sent(df_part):
    y = df_part.drop(['github', 'name'],axis=1)

    df_part = clean_context(df_part, 'context', 'context_tidy')
    df_part['redact_len'] = df_part['context'].apply(lambda x: count_redact(x))
    df_part.to_csv('test.tsv', sep="\t", index=False)

    X = df_part.drop(['github', 'name', 'context'],axis=1)
    return X, y

def clean_context(df, col, tidy_col):
    stop = stopwords.words('english')
    #df[tidy_col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # removes stop words
    df[tidy_col] = df[col].apply(lambda x: x.replace('\u2588','')) # removes redacted text
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

def load_data():
    data = []

    # loads data from tsv and separates based on data type (train, test, valid)
    with open('resources/data_handcleaned.tsv') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        
        # printing data line by line
        for line in tsv_file:
            data.append(line)

    # turns lists of lists into pandas df
    data_df = pd.DataFrame(data, columns = ['github', 'type', 'name', 'context'])

    return data_df

def main():
    # for training
    data_df = load_data()
    X, y = process_sent(data_df)

    X_train = (X.loc[X['type'] == 'training']).values.tolist()
    y_train = (y.loc[y['type'] == 'training']).values.tolist()

    X_valid = (X.loc[X['type'] == 'validation']).values.tolist()
    y_valid = (y.loc[y['type'] == 'validation']).values.tolist()

    X_test = (X.loc[X['type'] == 'testing']).values.tolist()
    y_test = (y.loc[y['type'] == 'testing']).values.tolist()

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    clf.predict(X_valid)
    clf.score(X_valid, y_valid)

if __name__ == "__main__":
    main()