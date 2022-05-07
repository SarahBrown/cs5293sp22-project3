from project3_functions import dataset
import project3

import pandas as pd
import numpy as np

def test_cleandata():
    use_saved_data = False
    save_data = False
    local_tsv = True
    
    # loads dataframe from locally saved unredactor.tsv
    data_df = project3.load_data(local_tsv)
    data_df = data_df.head()

    # stores expected and actual col names in np arrays
    expected_col_names = np.asarray(['github', 'type', 'name', 'context'])
    actual_col_names = np.asarray(data_df.columns)

    # asserts that expected equals actual
    np.testing.assert_equal(expected_col_names,actual_col_names)

    with open('tests/test_X.tsv') as file:
        expected_X = pd.read_csv(file, sep='\t+',engine='python', names = ['type', 'no_stop', 'redact_len', 'most_freqword1', 'most_freqword2',
                                                                        'pos-4', 'pos-3', 'pos-2', 'pos-1', 'pos+1', 'pos+2', 'pos+3', 'pos+4'])

    with open('tests/test_y.tsv') as file:
        expected_y = pd.read_csv(file, sep='\t+',engine='python', names = ['type', 'name'])

    expected_X = expected_X.to_numpy()
    expected_y = expected_y.to_numpy()

    actual_X, actual_y = dataset.process_sent(data_df,save_data)
    actual_X = actual_X.to_numpy()
    actual_y = actual_y.to_numpy()

    np.testing.assert_equal(expected_X,actual_X)
    np.testing.assert_equal(expected_y,actual_y)