import re
import string
import spacy


modComm = "tes&#100t\ning!!! string!! &#100"
index = 0
number = 'None'
asciiChar = ''
flag = False
code = '&#'
index = 0
number = 'None'
asciiChar = ''
flag = False
code = '&#'
modComm = modComm.replace('\n', ' ')
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
print (modComm)
print (string.punctuation)



index = 0
length = len(modComm)
consecPunctuation = False

#Go through every char in the string
while (index < length):
    #Found non apostrophe punctuation
    if (modComm[index] in string.punctuation and modComm[index] != "'" and modComm[index] != "-"):
        #Punctuation usually has a space after it or if it's consecutive punctuations
        if(index + 1 < length and (modComm[index + 1] == " " or modComm[index + 1] in string.punctuation) and consecPunctuation == False):
            length = length + 1
            #Place a space before the punctuation
            modComm = modComm[:index] + " " + modComm[index:]
            consecPunctuation = True
            
        #If this is the end of the string, there will be no space after it
        elif(index + 1 > length):
            length = length + 1
            #Place a space before the punctuation
            modComm = modComm[:index] + " " + modComm[index:]
    elif(consecPunctuation == True):
        consecPunctuation = False
    
    index = index + 1

print(modComm)

nlp = spacy.load('en', disable=['parser', 'ner'])
utt = nlp(u"I also know the best words")
for token in utt:
    #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    print(token.text, token.lemma_, str(token.pos_), token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)