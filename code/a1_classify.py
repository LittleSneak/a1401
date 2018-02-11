from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
import os

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    rightSum = 0
    totalSum = 0
    for x in range(0, len(C)):
        for y in range(0, len(C[0])):
            if(x == y):
                rightSum = rightSum + C[x][y]
            totalSum = totalSum + C[x][y]
    print(rightSum, totalSum)
    return rightSum / totalSum

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    print ('TODO')

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    print ('TODO')
    

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
    data = np.load(filename)['arr_0']
    X = data[:, :173]
    Y = data[:, 173:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    #Run all 5 classifiers
    #Linear SVC. Using LinearSVC instead of SVC since it is much faster
    print("Processing Linear")
    linear = LinearSVC()
    linear.fit(X_train, y_train.ravel())
    predictions1 = linear.predict(X_test)
    
    #Radial basis function, gamma = 2
    print("Processing radial basis")
    rb = SVC(kernel = 'rbf', gamma = 2)
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
    
    print(len(X), len(X_test), len(X_train))
    print(accuracy(linearCM))
    print(linearCM)
    
    #return (X_train, X_test, y_train, y_test, iBest)


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
    print('TODO Section 3.2')

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
    print('TODO Section 3.3')

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    class31(args.input)
