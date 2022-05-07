import re
from collections import Counter
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()

def process_sent(df_part,save_data):
    """Function to process dataset, clean context, and extract features."""

    print("Cleaning and sorting data...")

    y = df_part.drop(['context','github'],axis=1)
    y['name'] = y['name'].apply(lambda x: normalize_names(x))
    df_part = clean_context(df_part)
    df_part['no_stop'] =  df_part['context_tidy'].apply(lambda x: ' '.join([word for word in remove_stop(x)])) # removes stop words
    df_part = count_list(df_part)
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

def count_list(df_part):
    df_part['count_list'] = df_part['no_stop'].apply(lambda x: count_word(x))
    df_part['most_freqword1'] = df_part['count_list'].apply(lambda x: x[0][0])
    df_part['freq_count1'] = df_part['count_list'].apply(lambda x: x[0][1])
    df_part['most_freqword2'] = df_part['count_list'].apply(lambda x: x[1][0])
    df_part['freq_count2'] = df_part['count_list'].apply(lambda x: x[1][1])

    return df_part

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
