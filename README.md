# cs5293sp22-project3

### Author: Sarah Brown

# Directions to Install and Use Package
To download and use package, follow the steps below:

1. git clone https://github.com/SarahBrown/cs5293sp22-project3.git
2. cd cs5293sp22-project3/
3. pipenv install
4. Run via pipenv with one of the following example commands:
```pipenv run python project3.py```

Setting **use_saved_data** to **True** in the main function of project3.py will load and use the following files: precleaned_X.tsv and precleaned_y.tsv. This will not only use local data instead of downloading from github, but also skips dataset cleaning and feature extraction. This cuts down on runtime significantly. 

use_saved_data on **line 124** of project3.py should be set to True if there are any issues due to runtime during the "Cleaning and sorting data..." portion of running project3.py. Runtime of this portion of the code took about 1.5 minutes on my machine. Total project3.py runtime was about 2 minutes.

# Web or External Libraries
For this project I used several packages from the standard library and some external libraries. These included argparse, counter, glob, io, import, pickle, random, re, and urllib.request. In addition, the following external libraries were used pandas, spacy (and specifically en_core_web_lg), sklearn

# Assumptions Made and Known Bugs
The main assumption made is that due to the dataset used in this project, it is not possible to create a model with a high accuracy. This is due to the fact that any dataset that we train a model with, there will be many names in the testing and validation datasets that do not exist in the training model. Due to the fact that they do not exist in the training dataset, there will be no way to predict them correctly.

While running this code with the presaved data in the resources folder, there are no known bugs. However, with new pull requests being made to the unredactor.tsv file, new bugs may be introduced that are not handled. The presaved data can be run by setting the boolean **use_saved_data** to **True** in the main function of project3.py.

Setting **use_saved_data** to **True** in the main function of project3.py will load and use the following files: precleaned_X.tsv and precleaned_y.tsv.

# Functions and Approach to Development
By creating a different dataset separate from the class-sourced dataset, I was able to get scores of at least 4% (higher might have been possible with more possible). However, this method was very unwieldy and resulted in memory issues. As a result, I stuck to using the class-sourced dataset, even though this resulted in lower scores. 

The functions to implement this project were spread out across 3 python files. Project3.py in the main file structure and Dataset.py and Makedata.py in the project3_functions folder. Makedata.py contains functions adapted from 

### Project3.py
The main functionality of project 3 are in this python file. Using extracted features, a random forest classifier is trained to predict the redacted names. 

#### def load_data():

#### def train_model(X, y, save_model):

#### def load_model():

#### def main():

### Dataset.py
Stuff here about file

#### def process_sent(df_part,save_data):

#### def normalize_names(name):

#### def remove_stop(str):

#### def count_list(df_part):

#### def count_word(no_stop):

#### def count_redact(str):

#### def clean_context(df):

#### def pos_window(df_part):

#### def make_pos_window(context):

#### def how_much_data_overlap(y_train, y_test, y_valid, pred_valid, pred_test):

#### Makedata.py
Makedata.py contains functions adapted from my project1 code to extract training, testing, and validation from the imdb dataset provided.

# Tests
Tests are performed with PyTest and local data. Tests are also set up with Github Actions and PyTest to run automatically when code is pushed to the repository.

Tests are split across three files: test_cleandata.py and test_trainsaved.py.

## test_cleandata.py

## test_trainsaved.py