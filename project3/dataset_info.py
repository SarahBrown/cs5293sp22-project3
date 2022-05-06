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

def print_predicts(clf, X_valid, y_valid, X_test, y_test, y_train):
    predicts = clf.predict(X_valid)
    for i in range(len(predicts)):
        if (y_train.count(y_valid[i]) >= 1):
            #if (y_valid[i] == predicts[i]):
                print("Valid:",i)
                print(f'Data says:  {y_valid[i]}\nModel says: {predicts[i]}') #{X_valid[i]}\n
                print(f'Was y_valid[i] in train? {y_valid[i] in y_train}. Count: {y_train.count(y_valid[i])}')
 
    predicts = clf.predict(X_test)
    for i in range(len(predicts)):
        if (y_train.count(y_test[i]) >= 1):
            #if (y_test[i] == predicts[i]):
                print("Test:",i)
                print(f'Data says:  {y_test[i]}\nModel says: {predicts[i]}') #{X_valid[i]}\n
                print(f'Was y_test[i] in train? {y_test[i] in y_train}. Count: {y_train.count(y_test[i])}')
