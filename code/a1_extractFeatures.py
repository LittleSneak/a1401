import numpy as np
import sys
import argparse
import os
import json
import re
import string
import csv

#Lists containing words from word lists
#Also ensure it's all lower case
firstPerson = []
with open('/u/cs401/Wordlists/First-person') as fp:
    for line in fp:
        #Remove the \n char at the end
        line = line[:-1]
        firstPerson.append(line.lower())
        
secondPerson = []
with open('/u/cs401/Wordlists/Second-person') as fp:
    for line in fp:
        #Remove the \n char at the end
        line = line[:-1]
        secondPerson.append(line.lower())
        
thirdPerson = []
with open('/u/cs401/Wordlists/Third-person') as fp:
    for line in fp:
        #Remove the \n char at the end
        line = line[:-1]
        thirdPerson.append(line.lower())
    
conjunct = []
with open('/u/cs401/Wordlists/Conjunct') as fp:
    for line in fp:
        #Remove the \n char at the end
        line = line[:-1]
        conjunct.append(line.lower())
        
slang = []
with open('/u/cs401/Wordlists/Slang') as fp:
    for line in fp:
        #Remove the \n char at the end
        line = line[:-1]
        slang.append(line.lower())
        
#Build a dictionary for the Ratings_Warriner_et_al csv
ratings = {}
with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv') as file:
    reader = csv.reader(file, delimiter="\t")
    for i, line in enumerate(reader):
        items = line[0].split(",")
        ratings[items[1]] = items
        
#Build a dictionary for the BristolNorms+GilhoolyLogie csv
bristolNorms = {}
with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv') as file:
    reader = csv.reader(file, delimiter="\t")
    for i, line in enumerate(reader):
        items = line[0].split(",")
        bristolNorms[items[1]] = items


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    
    features = np.zeros((1, 173 + 1))
    #First find number of sentences
    index = 0
    sentences = 0
    
    #first step is to split words from tags
    taggedWords = re.findall(r"\S+\n*", comment)
    #List of all tagless words
    words = []
    #List of all tags
    tags = []
    for word in taggedWords:
        index = -1
        while(word[index] != "/"):
            index = index - 1
        words.append(word[:index + len(word)])
        tags.append(word[index + len(word):])
    
    #Iterate through all words and determine which feature it belongs to
    #Also use this to build up features 18-29
    
    #Total number of sentences
    sentences = 1
    numTokens = len(words)
    
    #For finding avg token length
    totalChars = 0
    index = 0
    
    #Going to state
    gtState = 0
    
    #Lists for features 18-31
    aoaList = []
    imgList = []
    famList = []
    vmeansumList = []
    ameansumList = []
    dmeansumList = []
    
    #Check every feature one by one for each word
    for word in words:
        #Feature 1
        if(word in firstPerson):
            features[0][0] = features[0][0] + 1
            
        #Feature 2
        if(word in secondPerson):
            features[0][1] = features[0][1] + 1
            
        #Feature 3
        if(word in thirdPerson):
            features[0][2] = features[0][2] + 1
            
        #Feature 4 conjunctions
        if(word in conjunct):
            features[0][3] = features[0][3] + 1
            
        #Feature 5 past tense verbs
        if(tags[index][1:5] == "VBD"):
            features[0][4] = features[0][4] + 1
            
        #Feature 6 future tense
        #States for finding "Going to + VB"
        if(word == "going"):
            gtState = 1
        elif(word == "to" and gtState == 1):
            gtState = 2
        elif(len(tags[index]) > 4 and tags[index][1:4] == "VB" and gtState == 2):
            gtState = 0
            features[0][5] = features[0][5] + 1
        else:
            gtState = 0
        if(word == "'ll" or word == "gonna" or word == "will"):
            features[0][5] = features[0][5] + 1
            
        #Feature 7 commas
        if(word == ","):
            features[0][6] = features[0][6] + 1
            
        #Feature 8 multi-character punctuation token
        #All words with greater than 1 length that are all punctuation
        if(len(word) > 1):
            x = 0
            while (x < len(word) and word[x] in string.punctuation):
                x = x + 1
            if(x == len(word)):
                features[0][7] = features[0][7] + 1
                
        #Feature 9 common nouns
        if(tags[index][1:4] == "NN" or tags[index][1:5] == "NNS"):
            features[0][8] = features[0][8] + 1
            
        #Feature 10 proper nouns
        elif(tags[index][1:5] == "NNP" or tags[index][1:6] == "NNPS"):
            features[0][9] = features[0][9] + 1
        
        #Feature 11 Adverbs
        elif(tags[index][1:5] == "RBR" or tags[index][1:5] == "RBS" or tags[index][1:4] == "RB"):
            features[0][10] = features[0][10] + 1
            
        #Feature 12 wh- words
        elif(tags[index][1:5] == "WDT" or tags[index][1:5] == "WRB" or tags[index][1:4] == "WP" or tags[index][1:5] == "WP$"):
            features[0][11] = features[0][11] + 1
            
        #Feature 13 Slang acronyms
        if(word in slang):
            features[0][12] = features[0][12] + 1
            
        #Feature 14 words in uppercase with len >= 3    
        if(len(word) >= 3 and word.isupper()):
            features[0][13] = features[0][13] + 1
        
        #Feature 16, total number of characters not including punctuation
        if(not word in string.punctuation):
            totalChars = totalChars + len(word)
            
        #Feature 17, total number of sentences
        if(tags[index][-1] == '\n'):
            sentences = sentences + 1
            
        #Feature 18, 21 AoA
        if(word in bristolNorms):
            aoaList.append(float(bristolNorms[word][3]))
        
        #Feature 19, 22 IMG
        if(word in bristolNorms):
            imgList.append(float(bristolNorms[word][4]))
            
        #Feature 20, 23 FAM
        if(word in bristolNorms):
            famList.append(float(bristolNorms[word][5]))
        
        #Feature 24, 27 V.Mean.Sum
        if(word in ratings):
            vmeansumList.append(float(ratings[word][2]))
        
        #Feature 25, 28 A.Mean.Sum
        if(word in ratings):
            ameansumList.append(float(ratings[word][5]))
        
        #Feature 26, 29 D.Mean.Sum
        if(word in ratings):
            dmeansumList.append(float(ratings[word][8]))
        
        index = index + 1
    
    #Put in the obtained values for the features
    features[0][14] = numTokens / sentences
    if(numTokens > 0):
        features[0][15] = totalChars / numTokens
    features[0][16] = sentences
    
    #Place the final features with averages and standard deviations
    if(len(aoaList) > 0):
        features[0][17] = np.mean(aoaList)
        features[0][20] = np.std(aoaList)
    if(len(imgList) > 0):
        features[0][18] = np.mean(imgList)
        features[0][21] = np.std(imgList)
    if(len(famList) > 0):
        features[0][19] = np.mean(famList)
        features[0][22] = np.std(famList)        
    if(len(vmeansumList) > 0):
        features[0][23] = np.mean(vmeansumList)
        features[0][26] = np.std(vmeansumList)
    if(len(ameansumList) > 0):
        features[0][24] = np.mean(ameansumList)
        features[0][27] = np.std(ameansumList)
    if(len(dmeansumList) > 0):
        features[0][25] = np.mean(dmeansumList)
        features[0][28] = np.std(dmeansumList)
    
    return features

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here
    #Go through every kson object and pass the body to extract1
    for index in range(0, len(data)):
        line = json.loads(data[index])
        feats[index] = extract1(line['body'])[0]
        
    #Do final features
    #Extract ID and put them into a dict
    
    #Dicts are ID to index
    idA = {}
    idC = {}
    idL = {}
    idR = {}
    with open ("/u/cs401/A1/feats/Alt_IDs.txt") as file:
        index = 0
        for line in file:
            idA[line[:-1]] = index
            index = index + 1
    with open ("/u/cs401/A1/feats/Center_IDs.txt") as file:
        index = 0
        for line in file:
            idC[line[:-1]] = index
            index = index + 1
    with open ("/u/cs401/A1/feats/Left_IDs.txt") as file:
        index = 0
        for line in file:
            idL[line[:-1]] = index
            index = index + 1
    with open ("/u/cs401/A1/feats/Right_IDs.txt") as file:
        index = 0
        for line in file:
            idR[line[:-1]] = index
            index = index + 1
    
    #Load all the npy files
    dataA = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
    dataC = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
    dataL = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
    dataR = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")
    
    #Go through every comment one by one
    for index in range(0, len(data)):
        line = json.loads(data[index])
        #Check which file it's from and act accordingly
        if(line['cat'] == "Alt"):
            idx = idA[line['id']]
            #Go through every feature and add them
            for index2 in range(0, 144):
                feats[index][29 + index2] = dataA[idx][index2]
			feats[index][173] = 3
                
        #Repeat for other 3 files
        elif(line['cat'] == "Center"):
            idx = idC[line['id']]
            for index2 in range(0, 144):
                feats[index][29 + index2] = dataC[idx][index2]
			feats[index][173] = 1
        
        elif(line['cat'] == "Left"):
            idx = idL[line['id']]
            for index2 in range(0, 144):
                feats[index][29 + index2] = dataL[idx][index2]
			feats[index][173] = 0
                
        elif(line['cat'] == "Right"):
            idx = idR[line['id']]
            for index2 in range(0, 144):
                feats[index][29 + index2] = dataR[idx][index2]
			feats[index][173] = 2
			
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

