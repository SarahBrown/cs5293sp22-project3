# cs5293sp22-project3

### Author: Sarah Brown

# Directions to Install and Use Package
To download and use package, follow the steps below:

1. git clone https://github.com/SarahBrown/cs5293sp22-project3.git
2. cd cs5293sp22-project3/
3. pipenv install
4. Run via pipenv with one of the following example commands:

# Web or External Libraries
For this project I used several packages from the standard library and some external libraries. These included argparse, json, os, and sys. In addition, the external librarie fuzzywuzzy was imported for string comparisons. Due to fuzzywuzzy causing a warning (Using slow pure-python SequenceMatcher), python-Levenshtein was also added to the Pipfile.

# Functions and Approach to Development

## Functions

### Project3.py
Stuff here about file

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
Stuff here about file

#### def redact_names(input_files, training_set_to_submit):
         
#### def redact_per_sent(inp):

#### def char_len(text):

#### def redact(text, start_char, end_char):

#### def add_arguments():

#### def get_inputfiles(input_glob):

#### def main():

# Assumptions Made and Known Bugs

# Tests
Tests are performed with PyTest and local data. Tests are also set up with Github Actions and PyTest to run automatically when code is pushed to the repository.

