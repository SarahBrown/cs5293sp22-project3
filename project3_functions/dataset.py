import re
from collections import Counter
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()

def process_sent(df_part,save_data):
    """Function to process dataset, clean context, and extract features."""
    print("Cleaning and sorting data...")
    # separates y off and normalizes names
    y = df_part.drop(['context','github'],axis=1)
    y['name'] = y['name'].apply(lambda x: normalize_names(x))

    # cleans context and removes stop words
    df_part = clean_context(df_part)
    df_part['no_stop'] =  df_part['context_tidy'].apply(lambda x: ' '.join([word for word in remove_stop(x)]))

    # extracts features
    df_part['redact_len'] = df_part['context'].apply(lambda x: count_redact(x)) 
    df_part = count_list(df_part)
    df_part = pos_window(df_part)

    X = df_part.drop(['name', 'context','context_tidy','pos_window','github'],axis=1)
    if (save_data):
        X.to_csv('resources/precleaned_X.tsv', sep="\t", index=False, header=False)
        y.to_csv('resources/precleaned_y.tsv', sep="\t", index=False, header=False)
    print("Data cleaned.")
    return X, y

def normalize_names(name):
    """Function to normalize names."""
    # removes any 's and splits names
    name = re.sub(r'\'s','', name)
    names = name.split(" ")

    # capitalizes each name and recombines
    new_name = ""
    for n in names:
        n = n.capitalize()
        new_name += n + " " 

    new_name = new_name[:-1] # removes trailing space
    return new_name

def clean_context(df):
    """Function to clean context string."""
    df['context_tidy'] = df['context'].apply(lambda x: re.sub(r'(\w+)n\'t', r'\g<1>' + " not", x)) # expands contractions
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('\u2588+','REDACTED',x)) # removes redacted text and replaces it with REDACTED
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('[^a-zA-Z \' \"]', ' ', x)) # removes punctuation
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub('[\'\"]', '', x)) # removes punctuation
    df['context_tidy'] = df['context_tidy'].apply(lambda x: re.sub(' +', ' ', x)) # replaces extra spaces

    return df

def remove_stop(str):
    """Function to remove stopwords and add as a feature."""
    doc = nlp(str)
    return [token.text for token in doc if (not token.is_stop and not token.is_punct)]

def count_list(df_part):
    """Splits results from count_word."""
    # uses count_word function to get two most frequent words and their counts
    df_part['count_list'] = df_part['no_stop'].apply(lambda x: count_word(x))
    # splits into different features
    df_part['most_freqword1'] = df_part['count_list'].apply(lambda x: x[0])
    df_part['most_freqword2'] = df_part['count_list'].apply(lambda x: x[1])
    df_part = df_part.drop(['count_list'],axis=1)

    return df_part

def count_word(no_stop):
    """Function to count the two most frequent words and return to be applied to dataframe."""
    # removes redacted filler from string
    no_stop = re.sub(' +', ' ', no_stop)
    no_stop = re.sub('REDACTED', '', no_stop)

    # creates a spacy nlp document
    doc = nlp(no_stop)

    # adds nouns to list
    words = [token.text for token in doc if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN")]

    # adds non-nouns if list length is not greater than 1 or 0
    if (len(words) == 1):
        more_words = [token.text for token in doc if (not token.is_stop and not token.is_punct)]
        for i in more_words:
            words.append(i)
    if (len(words) == 0):
        words = [token.text for token in doc if (not token.is_stop and not token.is_punct)]
        
    # counts words and finds 2 most common words
    word_freq = Counter(words)
    common_words = word_freq.most_common(2)

    list_common = []
    for i in common_words:
        list_common.append(f'{i[0]}_{i[1]}')

     # if length of most common is less than 2, add Empty with count 0
    while ((len(list_common) < 2)):
        list_common.append(f'Empty_0')

    return(list_common)

def count_redact(str):
    """Function to count length of redacted text."""
    count = 0
    for i in str:
        if (i == '\u2588'):
            count += 1

    return count

def pos_window(df_part):
    """Function to apply pos window to dataframe from make_pos_window."""
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
    """Function that makes a pos window +/- 4 indexes on either side of the redacted text."""
    doc = nlp(context)
    pos = ["Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty"]

    redact_index = [x for x in range(len(doc)) if doc[x].text == "REDACTED"] # gets list of elements matching REDACTED
    if (len(redact_index) > 0):
        redact_index = redact_index[0] # gets index of redacted string
    else:
        return pos

    # extracts text and pos and combines the two
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

def how_much_data_overlap(y_train, y_test, y_valid, pred_valid, pred_test):
    """Function to count overlap between training, testing, and validation datasets."""
    y_train_set = set()
    for i in y_train:
        y_train_set.add(i)

    y_test_set = set()
    test_correct = 0
    for i in range(len(y_test)):
        y_test_set.add(y_test[i])

        if (y_test[i] == pred_test[i]):
            test_correct += 1

    y_valid_set = set()
    valid_correct = 0
    for i in range(len(y_valid)):
        y_valid_set.add(y_valid[i])

        if (y_valid[i] == pred_valid[i]):
            valid_correct += 1

    test_total = 0
    valid_total = 0

    for i in y_test_set:
        test_total += y_train.count(i)

    for i in y_valid_set:
        valid_total += y_train.count(i)

    max_valid = round(valid_total/len(pred_valid),5)
    max_test = round(test_total/len(pred_test),5)
    my_valid = round(valid_correct/len(pred_valid),5)
    my_test = round(test_correct/len(pred_test),5)

    print(f'Max valid: {max_valid}\nMax test: {max_test}\nMy valid: {my_valid}\nMy test: {my_test}')
