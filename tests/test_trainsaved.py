import project3
import numpy as np

def test_trainsaved():
    use_saved_data = True
    save_data = False
    local_tsv = False
    save_model = False

    """Tests that presaved data was loaded properly by checking column names."""
    # loads X and y from locally saved precleaned_X and precleaned_y
    X, y = project3.load_dataset(use_saved_data, save_data, local_tsv)
    colX_names = ['type', 'no_stop', 'redact_len', 'most_freqword1', 'most_freqword2','pos-4', 'pos-3', 'pos-2', 'pos-1', 'pos+1', 'pos+2', 'pos+3', 'pos+4']
    coly_names = ['type', 'name']

    # stores expected and actual col names in np arrays
    expected_colX_names = np.asarray(colX_names)
    expected_coly_names = np.asarray(coly_names)
    actual_colX_names = np.asarray(X.columns)
    actual_coly_names = np.asarray(y.columns)

    # asserts that expected equals actual
    np.testing.assert_equal(expected_colX_names,actual_colX_names)
    np.testing.assert_equal(expected_coly_names,actual_coly_names)

    """Tests that scores are as expected."""
    # processes X,y into training, testing, and validation sets
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = project3.make_dict(X, y)
    # trains model
    clf = project3.train_model(X_train, y_train, save_model)

    # gets predicts for validation and testing for the model
    pred_valid = clf.predict(X_valid)
    pred_test = clf.predict(X_test)
    
    (prec_valid, prec_test), (recall_valid, recall_test), (f1_valid, f1_test) = project3.score_model(y_valid, y_test, pred_valid, pred_test)
    expected_prec_valid = 0.02029
    expected_prec_test = 0.01156
    expected_recall_valid = 0.02643
    expected_recall_test = 0.02572
    expected_f1_valid = 0.0199
    expected_f1_test = 0.0143

    # asserts that expected equals actual
    assert (expected_prec_valid == prec_valid)
    assert (expected_prec_test == prec_test)
    assert (expected_recall_valid == recall_valid)
    assert (expected_recall_test == recall_test)
    assert (expected_f1_valid == f1_valid)
    assert (expected_f1_test == f1_test)

    
