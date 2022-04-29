import argparse
import glob
import sys
import os

def main():
    # adds arguments, inputfiles, and performs redaction
    args = add_arguments()
    input_files = get_inputfiles(args.input)
    redact(args, input_files)

if __name__ == "__main__":
    main()

def add_arguments():
    """Function to add and parse arguments."""
    parser = argparse.ArgumentParser()        
    # get and return args
    args = parser.parse_args()
    return args

def get_inputfiles(input_glob):
    """Function to process glob and add input files."""
    files = sorted(glob.glob(input_glob)) # reads in glob files and sorts them alphabetically for ease of debugging
    file_list = []

    # opens each file in glob and creates a FileStats object to hold its details
    for f in files:
        file = open(f, "r") 
        new_file = FileStats.FileStats(f, file.read()) # creates FileStats obj
        file_list.append(new_file) # adds to list of files
        file.close()

    return file_list # returns list

def redact_names_files(args, input_file):
    