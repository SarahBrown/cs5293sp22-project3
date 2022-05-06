import re
import pickle
import numpy as np
import pandas as pd
import urllib.request
from collections import Counter
import spacy
import io
import en_core_web_lg
nlp = en_core_web_lg.load()

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score


def process_sent(df_part,save_data):
    print("Cleaning and sorting data...")
    y = df_part.drop(['context','github'],axis=1)
    y['name'] = y['name'].apply(lambda x: normalize_names(x))
    df_part = clean_context(df_part)
    df_part['no_stop'] =  df_part['context_tidy'].apply(lambda x: ' '.join([word for word in remove_stop(x)])) # removes stop words
    df_part['count_list'] = df_part['no_stop'].apply(lambda x: count_word(x))
    df_part['most_freqword1'] = df_part['count_list'].apply(lambda x: x[0][0])
    df_part['freq_count1'] = df_part['count_list'].apply(lambda x: x[0][1])
    df_part['most_freqword2'] = df_part['count_list'].apply(lambda x: x[1][0])
    df_part['freq_count2'] = df_part['count_list'].apply(lambda x: x[1][1])
    df_part = df_part.drop(['count_list'],axis=1)
    df_part = pos_window(df_part)

    X = df_part.drop(['name', 'context','context_tidy','pos_window','github'],axis=1)

    if (save_data):
        X.to_csv('resources/precleaned_X.tsv', sep="\t", index=False)
        y.to_csv('resources/precleaned_y.tsv', sep="\t", index=False)
    print("Data cleaned.")
    return X, y

def normalize_names(name):
    name = re.sub(r'\'s','', name)
    names = name.split(" ")

    new_name = ""
    for n in names:
        n.capitalize()
        new_name += n + " " 

    new_name = new_name[:-1]
    return new_name

def remove_stop(str):
    doc = nlp(str)
    return [token.text for token in doc if (not token.is_stop and not token.is_punct)]

def count_word(no_stop):
    no_stop = re.sub('REDACTED', '', no_stop)
    doc = nlp(no_stop)
    words = [token.text for token in doc if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN")]
    if (len(words) == 0):
        words = [token.text for token in doc if (not token.is_stop and not token.is_punct)]
    word_freq = Counter(words)
    common_words = word_freq.most_common(2)
    while ((len(common_words) < 2)):
        common_words.append(("NONE",0))

    return(common_words)

def count_redact(str):
    count = 0
    for i in str:
        if (i == '\u2588'):
            count += 1

    return count

def clean_context(df):
    df['redact_len'] = df['context'].apply(lambda x: count_redact(x))
    df['context_tidy'] = df['context'].apply(lambda x: re.sub(r'(\w+)n\'t', r'\g<1>' + " not", x))
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('\u2588+','REDACTED',x)) # removes redacted text
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('[^a-zA-Z \' \"]', ' ', x)) # removes punctuation
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('[\'\"]', '', x)) # removes punctuation
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub(' +', ' ', x)) # replaces extra spaces

    return df

def pos_window(df_part):
    df_part['pos_window'] = df_part['context_tidy'].apply(lambda x: make_pos_window(x))    

    df_part['pos-4'] = df_part['pos_window'].apply(lambda x: x[0])
    df_part['pos-3'] = df_part['pos_window'].apply(lambda x: x[1])
    df_part['pos-2'] = df_part['pos_window'].apply(lambda x: x[2])
    df_part['pos-1'] = df_part['pos_window'].apply(lambda x: x[3])
    df_part['pos+1'] = df_part['pos_window'].apply(lambda x: x[4])
    df_part['pos+2'] = df_part['pos_window'].apply(lambda x: x[5])
    df_part['pos+3'] = df_part['pos_window'].apply(lambda x: x[6])
    df_part['pos+4'] = df_part['pos_window'].apply(lambda x: x[7])

    return df_part

def make_pos_window(context):
    doc = nlp(context)
    pos = ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"]

    redact_index = [x for x in range(len(doc)) if doc[x].text == "REDACTED"] # gets list of elements matching REDACTED
    if (len(redact_index) > 0):
        redact_index = redact_index[0] # gets index of redacted string
    else:
        return pos

    if (redact_index - 4 > 0):
        pos[0] = doc[redact_index-4].text+"_"+doc[redact_index-4].pos_

    if (redact_index - 3 > 0):
        pos[1] = doc[redact_index-3].text+"_"+doc[redact_index-3].pos_

    if (redact_index - 2 > 0):
        pos[2] = doc[redact_index-2].text+"_"+doc[redact_index-2].pos_

    if (redact_index - 1 > 0):
        pos[3] = doc[redact_index-1].text+"_"+doc[redact_index-1].pos_

    if (redact_index + 1 < len(doc)):
        pos[4] = doc[redact_index+1].text+"_"+doc[redact_index+1].pos_

    if (redact_index + 2 < len(doc)):
        pos[5] = doc[redact_index+2].text+"_"+doc[redact_index+2].pos_

    if (redact_index + 3 < len(doc)):
        pos[6] = doc[redact_index+3].text+"_"+doc[redact_index+3].pos_
        
    if (redact_index + 4 < len(doc)):
        pos[7] = doc[redact_index+4].text+"_"+doc[redact_index+4].pos_

    return pos

def load_data():
    # headers for request so not to spam website
    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0"
    url = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"

    # loads data from tsv and loads into pandas df with column names
    try:
    # download from github
        print("Downloading data from github...")
        data = urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read()
        data = io.StringIO(data.decode('utf-8'))
        tsv_file = pd.read_csv(data, sep='\t+',engine='python', names = ['github', 'type', 'name', 'context'])

    except:
        # load local
        print("Download failed, loading local data.")
        with open('resources/unredactor.tsv') as file:
            tsv_file = pd.read_csv(file, sep='\t+',engine='python', names = ['github', 'type', 'name', 'context'])

    data_df = tsv_file.dropna()

    print("Data loaded.")
    return data_df

def train_model(X, y, save_model):
    print("Training model...")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X,y)
    print("Model is trained.")

    if (save_model):
        with open('model.pkl','wb') as f:
            pickle.dump(clf,f)
        print("model saved")

    acc_train = clf.score(X,y)
    print("Train Accuracy:", acc_train)

    return clf

def load_model():
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    return clf

def main():
    # boolean for use with peer review to avoid long runs if code takes a long time to run
    use_saved_data = False # set THIS boolean to true IF locally cleaning data does not work due to memory/timeout during peer review

    # booleans to use on my end to prepare saved data
    save_data = True # set to true when run locally on my machine to presave some cleaned data for use
    save_model = False # set to true to save model in pickled format

    if (not use_saved_data):
        # load and convert data
        data_df = load_data()
        X, y = process_sent(data_df, save_data)

    else:
        print("Loading presaved cleaned data...")
        with open('resources/precleaned_X.tsv') as file:
            X = pd.read_csv(file, sep='\t+',engine='python', names = ['type','redact_len','no_stop','most_freqword1','freq_count1',
                                    'most_freqword2','freq_count2','pos-4','pos-3','pos-2','pos-1','pos+1','pos+2','pos+3','pos+4'])
            #X = X.dropna()

        with open('resources/precleaned_y.tsv') as file:
            y = pd.read_csv(file, sep='\t+',engine='python', names = ['type', 'name'])
            #y = X.dropna()
        
        print("Data loaded.")

    dict = DictVectorizer(sparse=False)
    X_train = (X.loc[X['type'] == 'training']).drop(['type'], axis=1)
    print(X_train)
    X_test  = (X.loc[X['type'] == 'testing']).drop(['type'], axis=1)
    X_valid = (X.loc[X['type'] == 'validation']).drop(['type'], axis=1)

    y_train = list((y.loc[X['type'] == 'training']).drop(['type'], axis=1).to_numpy().flatten())
    y_test  = list((y.loc[X['type'] == 'testing']).drop(['type'], axis=1).to_numpy().flatten())
    y_valid = list((y.loc[X['type'] == 'validation']).drop(['type'], axis=1).to_numpy().flatten())

    X_train = dict.fit_transform(X_train.to_dict('records'))
    X_test = dict.transform(X_test.to_dict('records'))
    X_valid = dict.transform(X_valid.to_dict('records'))

    clf = train_model(X_train, y_train, save_model)

    pred_valid = clf.predict(X_valid)
    pred_test = clf.predict(X_test)

    prec_valid = round(precision_score(y_valid, pred_valid, average="weighted",zero_division=0),5)
    prec_test = round(precision_score(y_test, pred_test, average="weighted",zero_division=0),5)

    recall_valid = round(recall_score(y_valid, pred_valid, average="weighted",zero_division=0),5)
    recall_test = round(recall_score(y_test, pred_test, average="weighted",zero_division=0),5)

    f1_valid = round(f1_score(y_valid, pred_valid, average="weighted",zero_division=0),5)
    f1_test = round(f1_score(y_test, pred_test, average="weighted",zero_division=0),5)

    print(f'Validation Dataset:\n\tNumber of datapoints: {len(y_valid)}\n\tPrec: {prec_valid}\n\tRecall: {recall_valid}\n\tF1: {f1_valid}')
    print(f'Testing Dataset:\n\tNumber of datapoints: {len(y_test)}\n\tPrec: {prec_test}\n\tRecall: {recall_test}\n\tF1: {f1_test}')
    
if __name__ == "__main__":
    main()