import argparse
import glob
import re

import nltk
import spacy
import en_core_web_lg
import pandas as pd
import random

nlp = en_core_web_lg.load()

def redact_names(input_files, training_set_to_submit):
    """Function to redact names in input text file and generate test data."""
    # loops through each input file and finds names
    df = pd.DataFrame(columns =['github', 'type', 'name', 'context'])


    if (not training_set_to_submit):
        num_files = len(input_files)
        num_train = int(num_files*0.66)
        num_left = num_files - num_train
        num_valid = int(num_left*0.66)
        num_test = num_left-num_valid

        for input in range(0,num_train):
            inp = re.sub("<br /><br />", " ", input_files[input])
            redacted = redact_per_sent(inp)

            for i in range(len(redacted)):
                if (i % 3 == 0):
                    df.loc[len(df.index)] = ['SarahBrown', 'training', redacted[i][0], redacted[i][1]]
                elif (i % 3 == 1):
                    df.loc[len(df.index)] = ['SarahBrown', 'validation', redacted[i][0], redacted[i][1]]
                else:
                    df.loc[len(df.index)] = ['SarahBrown', 'testing', redacted[i][0], redacted[i][1]]
            
            if (input%100 == 0):
                print(f'{input}/{num_train}')

        print(f'{num_train} files done.')

        
        for input in range(num_train,num_train+num_valid):
            inp = re.sub("<br /><br />", " ", input_files[input])
            redacted = redact_per_sent(inp)

            for i in range(len(redacted)):
                df.loc[len(df.index)] = ['SarahBrown', 'training', redacted[i][0], redacted[i][1]]
            
            if (input%100 == 0):
                print(f'{input}/{num_valid}')

        print(f'{num_train+num_valid} files done.')

        for input in range(num_train+num_valid,num_train+num_valid+num_test):
            inp = re.sub("<br /><br />", " ", input_files[input])
            redacted = redact_per_sent(inp)

            for i in range(len(redacted)):
                df.loc[len(df.index)] = ['SarahBrown', 'training', redacted[i][0], redacted[i][1]]

            if (input%100 == 0):
                print(f'{input}/{num_test}')

        
        return df


    else:
        for inp in input_files:
            inp = re.sub("<br /><br />", " ", inp)
            redacted = redact_per_sent(inp)

            for redact in redacted:
                if (len(df.index) < 60):
                    df.loc[len(df.index)] = ['SarahBrown', 'training', redact[0], redact[1]]
                elif(len(df.index) < 110):
                    df.loc[len(df.index)] = ['SarahBrown', 'validation', redact[0], redact[1]]
                elif(len(df.index) < 130):
                    df.loc[len(df.index)] = ['SarahBrown', 'testing', redact[0], redact[1]]
                else:
                    return df
         
def redact_per_sent(inp):
    """Function to find and redact names in an input sentence."""
    redacted = []
    for sent in nlp(inp).sents:
        doc = nlp(sent.text)
        f = list(filter(lambda e: e.label_ in ["PERSON"], doc.ents))

        # filters to sentences with only one entity labeled person in them
        if (len(f) == 1):
            for ent in doc.ents:
                if (ent.label_ == "PERSON"):
                    redact_sent = redact(sent.text, ent.start_char, ent.start_char+len(ent.text)) # sent to redact, start char, end char

                    if (char_len(redact_sent) > 1000):
                        if (ent.start == 0):
                            redact_sent = redact_sent[:10]
                        elif (ent.start == len(redact_sent)-1):
                            redact_sent = redact_sent[10:-1]
                        else:
                            redact_sent = redact_sent[ent.start-5, ent.start+5]
                    
                    redacted.append([ent.text, redact_sent])

    return redacted

def char_len(text):
    """Function to count number of characters in a string."""
    total = 0

    for i in text:
        total += 1
    
    return total

def redact(text, start_char, end_char):
    """Function to replace character array with \u2588 characters."""
    redact_indexes = range(start_char, end_char)
    redacted = text

    for index in redact_indexes:
        redacted = redacted[:index] + "\u2588" + redacted[index+1:]

    return redacted

def add_arguments():
    """Function to add and parse arguments."""
    parser = argparse.ArgumentParser()     
    parser.add_argument("--input", type=str, required=True, 
                            help="Glob of local files stored in resources folder.")
   
    # get and return args
    args = parser.parse_args()
    return args

def get_inputfiles(input_glob):
    """Function to process glob and add input files."""
    files = glob.glob(input_glob) # reads in glob files
    file_list = []

    # opens each file in glob and adds text of file to a list
    for f in files:
        file = open(f, "r") 
        file_list.append(file.read()) # text to list
        file.close()

    random.shuffle(file_list)
    return file_list # returns list

def main():
    # adds arguments, inputfiles, and performs redaction
    args = add_arguments()
    input_files = get_inputfiles(args.input)

    training_set_to_submit = False

    df = redact_names(input_files, training_set_to_submit)
    print(f'Number of training: {len(df[df["type"] == "training"])}.\nNumber of valid: {len(df[df["type"] == "validation"])}.\nNumber of testing: {len(df[df["type"] == "testing"])}.')
    df.to_csv('resources/data.tsv', sep="\t", index=False)

if __name__ == "__main__":
    main()