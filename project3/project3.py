import re
import nltk

def redact_names(input_files):
    """Function to redact names in input text file."""
    # loops through each input file and finds names
    for inp in input_files:
        names = []
        nltk_names = []

        # looks for names via nltk
        sentences = nltk.sent_tokenize(inp.input_str)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]

        for tagged_sentence in sentences:
            for chunk in nltk.ne_chunk(tagged_sentence):
                if type(chunk) == nltk.tree.Tree:
                    if chunk.label() == 'PERSON':
                        found_name = chunk.leaves()[0][0]
                        nltk_names.append(found_name)                            

        # takes combined list and creates a set of names to search document via regex. also adds in nltk names
        regex_names = set()
        for nltk_n in nltk_names: # adds names found via nltk
            regex_names.add("\\b"+nltk_n+"\\b") 


        # searches input text string for matches to regex set
        name_matches = find_regex(regex_names, inp.input_str, False)
        inp.add_redact(name_matches, "names")

def find_regex(reg_set, input_str, ignore_case):
    """Function to find regex matches based on set of regex patterns."""
    matches = []
    for reg in reg_set:
        if (ignore_case): # ignores case
            iter = re.finditer(reg, input_str, re.IGNORECASE) # pattern, string, flags
        else: # looks for case differences
            iter = re.finditer(reg, input_str) # pattern, string, flags
        for match in iter:
            matches.append([match.group(), match.span()[0], match.span()[1]]) # string text, dx from start, string end char 

    return matches   