from project3_functions import dataset

import io
import pickle
import urllib.request
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score

def load_data(local_tsv):
    """Function to download tsv file from github."""
    # headers for url request
    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0"
    url = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"

    # loads unredactor.tsv if local_tsv is true
    if (local_tsv):
        print("Loading local data...")
        with open('resources/unredactor.tsv') as file: 
            tsv_file = pd.read_csv(file, sep='\t+',engine='python', names = ['github', 'type', 'name', 'context'])

    else:
        # loads data from tsv and loads into pandas df with column names
        try:
        # download from github
            print("Downloading data from github...")
            data = urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read() # gets file from github
            data = io.StringIO(data.decode('utf-8')) # decodes downloaded data
            tsv_file = pd.read_csv(data, sep='\t+',engine='python', names = ['github', 'type', 'name', 'context']) # makes pandas dataframe

        except:
            # loads local file if github data failed
            print("Download failed, loading local data.")
            with open('resources/unredactor.tsv') as file:
                tsv_file = pd.read_csv(file, sep='\t+',engine='python', names = ['github', 'type', 'name', 'context'])

    data_df = tsv_file.dropna() # drops any na

    print("Data loaded.")
    return data_df

def load_dataset(use_saved_data, save_data, local_tsv):
    """Function to download data from github or load local precleaned data."""
    # downloads data from github and cleans the dataset
    if (not use_saved_data):
        # load and convert data
        data_df = load_data(local_tsv)
        X, y = dataset.process_sent(data_df, save_data)

    # uses presaved and cleaned data to limit processing time
    # this option is only used if use_saved_data is set to True
    else:
        print("Loading presaved cleaned data...")
        with open('resources/precleaned_X.tsv') as file:
            X = pd.read_csv(file, sep='\t+',engine='python', names = ['type', 'no_stop', 'redact_len', 'most_freqword1', 'most_freqword2',
                                                                    'pos-4', 'pos-3', 'pos-2', 'pos-1', 'pos+1', 'pos+2', 'pos+3', 'pos+4'])

        with open('resources/precleaned_y.tsv') as file:
            y = pd.read_csv(file, sep='\t+',engine='python', names = ['type', 'name'])

        print("Data loaded.")

    return X, y

def make_dict(X, y):
    """Function to generate dictionary vectorizer and use it to make X, y pairs."""
    # separates X data into training, testing, and validation
    X_train = (X.loc[X['type'] == 'training']).drop(['type'], axis=1)
    X_test  = (X.loc[X['type'] == 'testing']).drop(['type'], axis=1)
    X_valid = (X.loc[X['type'] == 'validation']).drop(['type'], axis=1)

    # separates y data into training, testing, and validation and flattens it to a list
    y_train = list((y.loc[X['type'] == 'training']).drop(['type'], axis=1).to_numpy().flatten())
    y_test  = list((y.loc[X['type'] == 'testing']).drop(['type'], axis=1).to_numpy().flatten())
    y_valid = list((y.loc[X['type'] == 'validation']).drop(['type'], axis=1).to_numpy().flatten())

    # creates and uses dictionary vectorizer on X data from features extracted from the dataset
    dict = DictVectorizer(sparse=False)
    X_train = dict.fit_transform(X_train.to_dict('records'))
    X_test = dict.transform(X_test.to_dict('records'))
    X_valid = dict.transform(X_valid.to_dict('records'))

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def load_model():
    """Function to load and return a stored model."""
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    return clf

def score_model(y_valid, y_test, pred_valid, pred_test):
    """Function to calculate and return scores for model."""
    # calculates and prints out scores for model
    prec_valid = round(precision_score(y_valid, pred_valid, average="weighted",zero_division=0),5)
    prec_test = round(precision_score(y_test, pred_test, average="weighted",zero_division=0),5)

    recall_valid = round(recall_score(y_valid, pred_valid, average="weighted",zero_division=0),5)
    recall_test = round(recall_score(y_test, pred_test, average="weighted",zero_division=0),5)

    f1_valid = round(f1_score(y_valid, pred_valid, average="weighted",zero_division=0),5)
    f1_test = round(f1_score(y_test, pred_test, average="weighted",zero_division=0),5)

    return (prec_valid, prec_test), (recall_valid, recall_test), (f1_valid, f1_test)

def train_model(X, y, save_model):
    """Function to train and return a model to predict redacted names."""
    print("Training model...")
    # creates and trains a random forest classifier
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X,y)
    print("Model is trained.")

    # saves model if boolean is set to true
    if (save_model):
        with open('model.pkl','wb') as f:
            pickle.dump(clf,f)
        print("model saved")

    # calculates and prints training accuracy
    acc_train = clf.score(X,y)
    print("Train Accuracy:", acc_train)

    return clf

def main():
    """Main function to process training data and to predict validation and testing data."""
    # boolean for use with peer review to avoid long runs if code takes a long time to run
    use_saved_data = True # set THIS BOOLEAN to true IF locally cleaning data does not work due to memory/timeout during peer review

    # booleans to set when run locally on my machine to presave cleaned data for future use
    # do NOT change these
    save_data = False # saves prepocessed X and y tsv files
    save_model = False # saves pretrained model as a pickle file
    local_tsv = False # set equal to true to use local unredactor.tsv

    X, y = load_dataset(use_saved_data, save_data, local_tsv)

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = make_dict(X, y)

    # trains model
    clf = train_model(X_train, y_train, save_model)

    # gets predicts for validation and testing for the model
    pred_valid = clf.predict(X_valid)
    pred_test = clf.predict(X_test)
    
    (prec_valid, prec_test), (recall_valid, recall_test), (f1_valid, f1_test) = score_model(y_valid, y_test, pred_valid, pred_test)

    print(f'Validation Dataset:\n\tNumber of datapoints: {len(y_valid)}\n\tPrec: {prec_valid}\n\tRecall: {recall_valid}\n\tF1: {f1_valid}')
    print(f'Testing Dataset:\n\tNumber of datapoints: {len(y_test)}\n\tPrec: {prec_test}\n\tRecall: {recall_test}\n\tF1: {f1_test}')

if __name__ == "__main__":
    main()