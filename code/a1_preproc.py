import sys
import argparse
import os
import json
import re
import string
import spacy

indir = '/u/cs401/A1/data/';

nlp = spacy.load('en', disable=['parser', 'ner'])
#Contains the abbreviations from the abbreviations file
abbreviations = []
with open('/u/cs401/Wordlists/abbrev.english') as fp:
    for line in fp:
        #Remove the \n char at the end
        line = line[:-1]
        abbreviations.append(line)

#Contains the stopwords from the StopWords file
stopWords = []
with open('/u/cs401/Wordlists/StopWords') as fp:
    for line in fp:
        #Remove the \n char at the end
        line = line[:-1]
        stopWords.append(line)


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    alreadyTagged = False
    wordToLemma = {}
    #Remove new line characters
    if 1 in steps:
        modComm = comment.replace('\n', ' ')
        
    #Convert HTML codes to ASCII
    if 2 in steps:
        index = 0
        number = 'None'
        asciiChar = ''
        flag = False
        code = '&#'
        while index < len(modComm):
            if(modComm[index] == '&'):
                flag = True
            #Found a sequence of &#
            #Try to get a number
            elif(modComm[index] == '#' and flag):
                if(index + 3 < len(modComm) and modComm[index + 1 : index + 4].isdigit()):
                    number = modComm[index + 1 : index + 4]
                elif(index + 2 < len(modComm) and modComm[index + 1 : index + 3].isdigit()):
                    number = modComm[index + 1 : index + 3]
                elif(index + 1 < len(modComm) and modComm[index + 1].isdigit()):
                    number = modComm[index + 1]
                
                #Found an appropriate code, replace with ascii
                if(number != 'None'):
                    asciiChar = chr(int(number))
                    code = code + number
                    modComm = modComm.replace(code, asciiChar)
                    code = '&#'
                    number = 'None'
                    flag = False
                #Did not find a code
                else:
                    flag = False
            else:
                flag = False
            index = index + 1
                
    #Replace all tokens starting with http and www. with white space
    if 3 in steps:
        modComm = re.sub(r'http\S+', ' ', modComm)
        modComm = re.sub(r'www.\S+', ' ', modComm)
    #Split all punctuation into a separate token
    if 4 in steps:        
        words = modComm.split()
        newWords = []
        #Look at each word one at a time
        for word in words:
            #Word ends in non-period and non-apostrophe
            if(word[-1] in string.punctuation and word[-1] != "." and word[-1] != "'"):
                #Find first instance of non-punctuation
                index = 1
                while(index <= len(word) and word[-1 * index] in string.punctuation):
                    index = index + 1
                index = len(word) - index + 1
                newWords.append(word[:index] + " " + word[index:])
            #Word ends in period check for abbreviation
            elif(word[-1] == "."):
                if(word not in abbreviations):
                    #Find first instance of non-punctuation
                    index = 1
                    while(index <= len(word) and word[-1 * index] in string.punctuation):
                        index = index + 1
                    index = len(word) - index + 1
                    newWords.append(word[:index] + " " + word[index:])
                else:
                    newWords.append(word)
            else:
                newWords.append(word)
        modComm = ""
        for word in newWords:
            modComm = modComm + word + " "
                
        
        
    #Split clitics using whitespace
    if 5 in steps:
        index = 0
        #Go through all chars and find apostrophes
        for letter in modComm:
            if(letter == "'"):
                #First case, possessive plural
                #Check if the letter before the ' is an s then check if this is the end
                #of the word or comment
                if((modComm[index - 1] == "s") and (index + 1 == len(modComm) or (index + 1 < len(modComm) and (modComm[index + 1] == " ")))):
                    modComm = modComm[:index] + " " + modComm[index:]
                    index = index + 1
                #Case: 's
                elif(index + 1 < len(modComm) and modComm[index + 1] == 's'):
                    modComm = modComm[:index] + " " + modComm[index:]
                    index = index + 1
                #Case: n't
                elif(index + 1 < len(modComm) and modComm[index + 1] == 't' and modComm[index - 1] == 'n'):
                    modComm = modComm[:index - 1] + " " + modComm[index - 1:]
                    index = index + 1     
                #Case: 've
                elif(index + 2 < len(modComm) and modComm[index + 1] == 'v' and modComm[index + 2] == 'e'):
                    modComm = modComm[:index] + " " + modComm[index:]
                    index = index + 1
                #Case: 're
                elif(index + 2 < len(modComm) and modComm[index + 1] == 'r' and modComm[index + 2] == 'e'):
                    modComm = modComm[:index] + " " + modComm[index:]
                    index = index + 1
            index = index + 1
                    
    #Tag all words with spacy  
    if 6 in steps:
        alreadyTagged = True
        utt = nlp(modComm)
        modComm = ''
        for token in utt:
            if(token.text != " "):
                modComm = modComm + token.text + "/" + token.tag_ + " "
                wordToLemma[token.text + "/" + token.tag_] = token.lemma_ + "/" + token.tag_
            
    
    #Remove stop words
    if 7 in steps:
        words = modComm.split()
        
        #Build up a new list of words
        newWords = []
        
        for word in words:
            #Get the word without the tag
            index = 0
            while(word[index] != "/"):
                index = index + 1
            taglessWord = word[:index]
            #Only add the word if it's not a stopword
            if (not taglessWord in stopWords):
                newWords.append(word)
        modComm = ""
        for word in newWords:
            modComm = modComm + word + " "
            
    #Apply lemmatization
    if 8 in steps:
        #Remove tags from all words
        words = modComm.split()
        #Step 6 already run before, use tags from there
        if(alreadyTagged == True):
            modComm = ""
            for word in words:
                if(wordToLemma[word][0] != "-"):
                    modComm = modComm + wordToLemma[word] + " "
                else:
                    modComm = modComm + word + " "
        #Step 6 not run, need to make tags
        else:
            taglessComm = ""
            #Stores all tags in order
            for word in words:
                index = 0
                while(word[index] != "/"):
                    index = index + 1
                taglessComm = taglessComm + word[:index] + " "
            #Run spacy on comment to get lemmatizations
            utt = nlp(taglessComm)
            index = 0
            modComm = ""
            for token in utt:
                if(token.lemma_[0] != "-"):
                    modComm = modComm + token.lemma_ + "/" + token.tag_ + " "
                else:
                    modComm = modComm + token.text + "/" + token.tag_ + " "
                index = index + 1
        
    if 9 in steps:
        words = modComm.split()
        index = 0
        #Iterate through all words
        #Words tagged with /. are for end of sentence
        while(index < len(words)):
            #Found a sentence ending word, move forward until the last one (Handling consec punctuation)
            if(words[index][-1] == "."):
                while(index < len(words) and words[index][-1] == "."):
                    index = index + 1
                words[index - 1] = words[index - 1] + '\n'
                index = index - 1
            index = index + 1
        #Reconstruct sentence
        modComm = ''
        for word in words:
            modComm = modComm + word + " "
    if 10 in steps:
        words = re.findall(r"\S+\n*", modComm)
        modComm = ""
        #Go through all words and make each one lower case
        for word in words:
            index = -1
            #Split the tag from the word
            while(word[index] != "/"):
                index = index - 1
            tag = word[len(word) + index:]
            taglessWord = word[:len(word) + index]
            modComm = modComm + taglessWord.lower() + tag + " "
        
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print ("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            for loop in range(0, int(args.max)):
                # TODO: read those lines with something like `j = json.loads(line)`
                line = json.loads(data[(args.ID[0] + loop) % len(data)])
                # TODO: choose to retain fields from those lines that are relevant to you
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                newJsonObj = {}
                newJsonObj['id'] = line['id']
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                newJsonObj['cat'] = file
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # TODO: replace the 'body' field with the processed text
                newJsonObj['body'] = preproc1(line['body'])
                json_data = json.dumps(newJsonObj)
                # TODO: append the result to 'allOutput'
                allOutput.append(json_data)
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()
    print("done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print ("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
        
    main(args)
