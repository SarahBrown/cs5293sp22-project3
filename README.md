# cs5293sp22-project3

### Author: Sarah Brown

# Directions to Install and Use Package
To download and use package, follow the steps below:

1. git clone https://github.com/SarahBrown/cs5293sp22-project3.git
2. cd cs5293sp22-project3/
3. pipenv install
4. Run via pipenv with the following command:
```pipenv run python project3.py```

5. Pytests can be run with the following command:
```pipenv run python -m pytest```

Setting **use_saved_data** to **True** in the main function of project3.py will load and use the following files: precleaned_X.tsv and precleaned_y.tsv. This will not only use local data instead of downloading from github, but also skips dataset cleaning and feature extraction. This cuts down on runtime significantly. 

use_saved_data on **line 131** of project3.py should be set to True if there are any issues due to runtime during the "Cleaning and sorting data..." portion of running project3.py. Runtime of this portion of the code took about 1.5 minutes on my machine. Total project3.py runtime was about 2 minutes.

# Web or External Libraries
For this project I used several packages from the standard library and some external libraries. These included argparse, counter, glob, io, import, pickle, random, re, and urllib.request. In addition, the following external libraries were used pandas, spacy (and specifically en_core_web_lg), and sklearn.

# Assumptions Made and Known Bugs
The main assumption made is that due to the dataset used in this project, it is not possible to create a model with a high accuracy. This is due to the fact that any dataset that we train a model with, there will be many names in the testing and validation datasets that do not exist in the training dataset. Due to the fact that they do not exist in the training dataset, there will be no way to predict them correctly.

While running this code with the presaved data in the resources folder, there are no known bugs. However, with new pull requests being made to the unredactor.tsv file, new bugs may be introduced that are not currently handled. If this occurs, the presaved data can be run by setting the boolean **use_saved_data** to **True** in the main function of project3.py.

Setting **use_saved_data** to **True** in the main function of project3.py will load and use the following files: precleaned_X.tsv and precleaned_y.tsv.

# Functions and Approach to Development
By creating a different dataset separate from the class-sourced dataset, I was able to get scores of at least 4% (higher might have been possible with more experimentation). However, this method was very unwieldy and resulted in memory issues. As a result, I stuck to using the class-sourced dataset, even though this resulted in lower scores. 

The functions to implement this project were spread out across 3 python files. Project3.py in the main file structure and Dataset.py and Makedata.py in the project3_functions folder. Makedata.py contains functions adapted from project1 and is used to generate my dataset that I submitted.

### Project3.py
The main functionality of project 3 are in this python file. Using extracted features, a random forest classifier is trained to predict the redacted names. 

#### load_data(local_tsv):
The load_data function is used to load in the training, testing, and validation datasets. This is done in different parts. First, a boolean is passed into the function to indiciate whether the program should load the locally saved dataset. If the boolean is true, the dataset is loaded from 'resources/unredactor.tsv'

Otherwise, the dataset is downloaded from the latest version of the github unredactor.tsv file. This is executed in a try/except block. If the urllib request or following decoding fails to process the file from github, the except block will instead load the locally stored unredactor dataset.

#### load_dataset(use_saved_data, save_data, local_tsv):
The load_dataset function is used to select between using the raw dataset or the preprocessed dataset. If the raw dataset is used, the data is loaded from either github or a local file as described above and then the process_sent function is called from the dataset.py file. 

However, cleaning, organizing, and extracting features from the dataset takes a decent amount of time and to limit the risk of errors during the peer review process, a boolean was added to allow the use of preprocessed data. If this boolean, use_saved_data, is set to True, instead of processing the dataset and extracting features, those features will be loaded from precleaned_X.tsv and preceleaned_y.tsv respectively. This data was created locally on my machine during a run with finished code. This allows the user to skip the feature extraction process and go straight to training steps.

#### make_dict(X, y):
The make_dict function is used to change the X and y dataframes into separate X and y training, testing, and validation datasets. This is done by filtering the pandas dataframes and changing the y datasets to lists. In addition, a dictionary vectorizer is used on the extracted features from the X training set and is used to transform the X testing and validation sets as well. The resulting datasets are returned to use for model training and scoring.

#### train_model(X, y, save_model):
The train_model function is used to apply a random forest classifier to the training dataset. It is possible to pass a save_model boolean to this function to indicate that the model should be saved for future by pickling the data. In addition, the training accuracy is printed before returning the model for scoring with the testing and validation datasets.

#### load_model():
The load_model function is used to load any previously saved models by un-pickling them. This was originally used while working with non class-sourced datasets due to the size of the data and resulting model. This is not currently used as the saved models end up being too large for Github storage, but was left for reference.

#### score_model():
The score_model function takes in the y testing and validation datasets and their corresponding predictions according to the model created. It uses theses inputs to find the precision, recall, and f1 scores of the model. These scores are then returned to the main function and printed out.

### Dataset.py
This python file stores functions used to clean the dataset and extract features used for training. The cleaned data and features are formatted into a pandas dataframe for ease of use to create a model.

#### process_sent(df_part,save_data):
The process_sent function is used to streamline the processing of the unredactor.tsv dataset. It calls other functions from dataset.py and applies the changes to the passed in dataframe. This function is also used to save the stored precleaned datafiles (precleaned_X.tsv and precleaned_y.tsv) that can be selected to use for processing.

#### normalize_names(name):
The normalize_names function removes any 's from the name and normalizes capitalization. This condenses some of the y's by creating more overlap between datapoints.

#### clean_context(df):
The clean_context function is used to clean up the context string to make it easier to extract features from. This is done by expanding contractions, removing the \u2588 redaction blocks, removing punctuation, and removing any extra spaces.

#### remove_stop(str):
The remove_stop function is used to remove any stopwords from the cleaned context return by clean_context. This is done using Spacy's stopwords list. This is returned to be used as a feature.

#### count_list(df_part):
The count_list function is used to apply the results of the count_word function to the dataframe and drops the temporary count_list column. The columns that result from this are used as features.

#### count_word(no_stop):
The count_word function is used to find and count the two most frequent words from the cleaned context string without stopwords. 

#### count_redact(str):
The count_redact function is used to find the length of the redacted text in the context string. This is returned to be used as a feature.

#### pos_window(df_part):
The pos_window function is used to apply the results of the pos_window function to the dataframe. The columns that result from this are used as features.

#### make_pos_window(context):
The make_pos_window function is used to find the four words on either side of the redacted name. The words and their respective parts of speech are joined and returned for processing in the pos_window function and are added as features to the X dataframe. If there are any indexes that do not have a word, "EMPTY" is returned instead.

#### how_much_data_overlap(y_train, y_test, y_valid, pred_valid, pred_test):
The how_much_data_overlap function is used to count overlap between the y training, testing, and validation datasets. This function was used during development to determine how much overlap there was between training the model and using the model to see what scores could be expected. The results, due to the dataset used for this project, were very low. This function is not called while running the project, but is left for reference.

#### Makedata.py
Makedata.py contains functions adapted from my project1 code to extract training, testing, and validation from the imdb dataset provided.

# Tests
Tests are performed with PyTest and local data. Tests are also set up with Github Actions and PyTest to run automatically when code is pushed to the repository.

Tests are split across two files: test_cleandata.py and test_trainsaved.py.

## test_cleandata.py
This file contains tests that confirm that the dataset is loaded properly and the data is cleaned properly. 

This is done by loading via the created load_data() function with a locally stored unredactor.tsv file and confirming that the column names are as expected. This dataset is then shortned to only its head. By only passing 5 data entries instead of the full dataset, this allows for hand confirmation of test results. Next, the expected values are loaded from separate tsv files. The actual values gotten as a result of the function dataset.process_sent() and the expected values were then compared using numpy's testing function of assert_equal. This function allows comparison between numpy arrays.

## test_trainsaved.py
This file contains tests that confirm that the presaved and cleaned data loads properly and results in the expected model scores for the testing and validation datasets. 

This is done by first loading the extracted features from the presaved and cleaned data with the load_dataset() function. This is then confirmed to have worked properly by confirming that the column names for the X and y dataframes match the expected column names. This comparision is done with numpy's testing function assert_equal. The model is then trained and scored on the loaded data. The expected score values are then compared to the actual score values to confirm that the model loaded properly. 