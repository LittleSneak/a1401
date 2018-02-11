from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import f_classif
import csv
import numpy as np
import argparse
import sys
import os

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    #Sum of diagonal in matrix
    rightSum = 0
    #Sum of all values in matrix
    totalSum = 0
    for x in range(0, len(C)):
        for y in range(0, len(C[0])):
            if(x == y):
                rightSum = rightSum + C[x][y]
            totalSum = totalSum + C[x][y]
    if(totalSum != 0):
        return rightSum / totalSum
    else:
        return 0

def recall( C ):
    #row column
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    retList = []
    #A class belongs to a row
    #Correctly classified parts are in the diagonal
    for x in range(0, len(C)):
        rowSum = 0
        correct = 0        
        for y in range(0, len(C[0])):
            if(x == y):
                correct = correct + C[x][y]
            rowSum = rowSum + C[x][y]
        if(rowSum != 0):
            retList.append(correct / rowSum)
        else:
            retList.append(0)
    return retList
            

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    retList = []
    for y in range(0, len(C[0])):
        columnSum = 0
        correct = 0        
        for x in range(0, len(C)):
            if(x == y):
                correct = correct + C[x][y]
            columnSum = columnSum + C[x][y]
        if(columnSum != 0):
            retList.append(correct / columnSum)
        else:
            retList.append(0)
    return retList
             

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    #Load the npy from the file and put it through training test split
    print("Beginning 3.1")
    data = np.load(filename)['arr_0']
    X = data[:, :173]
    Y = data[:, 173:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, test_size=0.2)
    
    #Run all 5 classifiers
    #Linear SVC. Using LinearSVC instead of SVC since it is much faster
    print("Processing Linear")
    linear = LinearSVC(max_iter=10)
    linear.fit(X_train, y_train.ravel())
    predictions1 = linear.predict(X_test)
    
    #Radial basis function, gamma = 2
    print("Processing radial basis")
    rb = SVC(kernel = 'rbf', gamma = 2, max_iter=10)
    rb.fit(X_train, y_train.ravel())
    predictions2 = rb.predict(X_test)
    
    #Random Forest Classifier. max depth = 5 and 10 estimators
    print("Processing forest")
    forest = RandomForestClassifier(max_depth=5, n_estimators=10)
    forest.fit(X_train, y_train.ravel())
    predictions3 = forest.predict(X_test)
    
    #MLPClassifier with alpha = 0.05
    print("Processing MLP")
    mlp = MLPClassifier(alpha = 0.05)
    mlp.fit(X_train, y_train.ravel())
    predictions4 = mlp.predict(X_test)
    
    #AdaBoostClassifier with default
    print("Processing AdaBoost")
    adaboost = AdaBoostClassifier()
    adaboost.fit(X_train, y_train.ravel())
    predictions5 = adaboost.predict(X_test)
    
    #Obtain confusion matrices
    linearCM = confusion_matrix(y_test, predictions1)
    rbfCM = confusion_matrix(y_test, predictions2)
    forestCM = confusion_matrix(y_test, predictions3)
    mlpCM = confusion_matrix(y_test, predictions4)
    adaboostCM = confusion_matrix(y_test, predictions5)
    
    #Write to file
    with open('a1_3.1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        
        cms = [linearCM, rbfCM, forestCM, mlpCM, adaboostCM]
        
        #Iterate through CMs and write them
        #Keep track of accuracies for 3.2
        accList = []
        for index in range(0, 5):
            acc = accuracy(cms[index])
            
            row = [str(index + 1)]
            #Accuracy
            row.append(str(acc))
            
            newScore = recall(cms[index])
            for x in range(0, 4):
                row.append(str(newScore[x]))
                
            newScore = precision(cms[index])
            for x in range(0, 4):
                row.append(str(newScore[x]))
            
            for x in range(0, 4):
                for y in range(0, 4):
                    row.append(str(cms[index][x][y]))
            writer.writerow(row)
            accList.append(acc)
            
    #Find the best classifier
    maxIndex = 0
    maxAcc = 0
    for index in range(0, 5):
        if(accList[index] > maxAcc):
            maxIndex = index
            maxAcc = accList[index]

    return (X_train, X_test, y_train, y_test, maxIndex)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    #Obtain the best classifier to use
    print("Beginning 3.2")
    if(iBest == 0):
        classifier = linear = LinearSVC(max_iter=10)
        print("Linear chosen")
    elif(iBest == 1):
        classifier = rb = SVC(kernel = 'rbf', gamma = 2, max_iter = 10)
        print("Radial basis chosen")
    elif(iBest == 2):
        classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
        print("Random forest chosen")
    elif(iBest == 3):
        classifier = MLPClassifier(alpha = 0.05)
        print("MLP chosen")
    else:
        classifier = AdaBoostClassifier()
        print("Ada boost chosen")
    
    #Test each one
    train_sizes = [1000, 5000, 10000, 15000, 20000]
    accList = []
    for size in train_sizes:
        print("Processing size " + str(size))
        test_sz =  len(X_train) - size
        X_traint, X_testt, y_traint, y_testt = train_test_split(X_train, y_train, train_size=size, test_size = test_sz)
        #Keep the 1k train sizes for return
        if(size == 1000):
            X_1k = X_traint
            y_1k = y_traint
        #Perform fitting and accuracy calculations
        classifier.fit(X_traint, y_traint.ravel())
        accList.append(accuracy(confusion_matrix(y_testt, classifier.predict(X_testt))))
    
    
    #Write to a csv file
    with open('a1_3.2.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        row = []
        for item in accList:
            row.append(str(item))
        writer.writerow(row)

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    #Go through each k value
    print("Beginning 3.3")
    #Stores rows to write to csv
    rows = []
    for k in [5, 10, 20, 30, 40, 50]:
        selector = SelectKBest(f_classif, k)
        X_new32k = selector.fit_transform(X_train, y_train.ravel())
        pp = selector.pvalues_
        
        selector = SelectKBest(f_classif, k)
        X_new1k = selector.fit_transform(X_1k, y_1k.ravel())
        #Save X_new for 5 best features for step 2
        if(k == 5):
            best5_32k = X_new32k
            best5_1k = X_new1k
        #Only save pp for 32k
        row = [str(k)]
        print(len(pp))
        for pval in pp:
            row.append(str(pval))
        rows.append(row)
        
        
    #Obtain the best classifier to use
    if(i == 0):
        classifier = linear = LinearSVC(max_iter=10000)
        print("Linear chosen")
    elif(i == 1):
        classifier = rb = SVC(kernel = 'rbf', gamma = 2, max_iter = 10000)
        print("Radial basis chosen")
    elif(i == 2):
        classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
        print("Random forest chosen")
    elif(i == 3):
        classifier = MLPClassifier(alpha = 0.05)
        print("MLP chosen")
    else:
        classifier = AdaBoostClassifier()
        print("Ada boost chosen")
    
    #Train 5 best features for 1k
    accList = []
    
    """print("Running classifier for 1k")
    
    print(best5_1k)
    print(len(best5_1k))
    
    classifier.fit(best5_1k, y_1k.ravel())
    predictions1k = classifier.predict(X_test)
    cm1k = confusion_matrix(y_test, predictions1k)
    accList.append(str(accuracy(cm1k)))
    
    #Train 5 best features for 32k
    print("Running classifier for 32k")
    classifier.fit(best5_32k, y_train.ravel())
    predictions32k = classifier.predict(X_test)
    cm1k = confusion_matrix(y_test, predictions32k)
    accList.append(str(accuracy(cm1k)))
    
    rows.append(accList)
    print(accList)
    """
    #Write to the csv file
    with open('a1_3.3.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
        
        

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    results31 = class31(args.input)
    results32 = class32(results31[0], results31[1], results31[2], results31[3], results31[4])
    class33(results31[0], results31[1], results31[2], results31[3], results31[4], results32[0], results32[1])
    class34(args.input, results31[4])