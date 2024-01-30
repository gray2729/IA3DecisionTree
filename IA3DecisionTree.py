"""
File name:  IA3DecisionTree.py
Author:     Isaac Gray
Date:  	    03/22/2023
Class: 	    DSCI 440 ML
Assignment: IA 3
Purpose:    Builds decision stump and decision tree
"""
import math
from PrintTree import print_tree


class DecisionTreeNode:
    def __init__(self, testFeat):
        self.data = None
        self.label = None
        self.testNum = testFeat
        self.testValue = None
        self.left = None
        self.right = None
    

class DecisionTree:
    def __init__(self):
        self.root = None


    """
    Function:    buildTree
    Description: builds decision tree with a certain depth
    Input:       X - matrix holding the features
                 Y - vector holding the target training values 
                 wantedDepth - the depth of the tree
    Output:      None
    """
    def buildTree(self, X, Y, wantedDepth):
        self.root = DecisionTreeNode(None)
        self.buildBranch(X, Y, self.root, wantedDepth-1)
    
    
    """
    Function:    buildBranch
    Description: builds a node of the decision tree by determining
                 and using the optimal feature
    Input:       X - matrix holding the features
                 Y - vector holding the target training data 
                 curr_node - the current node that is being built
                 depth - remaining depth of the tree
    Output:      None
    """
    def buildBranch(self, X, Y, curr_node, depth):
        
        item1Count, item2Count, item1Label, item2Label = fCount(Y)

        if depth >= 1 and item2Count != 0:    
            curr_node.left = DecisionTreeNode(None)
            curr_node.right = DecisionTreeNode(None)
            
            maxBenefit = 0
            maxFeature = None
            
            for feature in range(len(X[0])):
                reduction = fBenefitOfSplit(X, Y, feature)
                if reduction > maxBenefit:
                    maxBenefit = reduction
                    maxFeature = feature
            
            curr_node.testNum = maxFeature
            curr_node.data = maxFeature+1
            curr_node.testValue = X[0][maxFeature]
            print("Feature: ", maxFeature+1)
            print("Information gained: ", maxBenefit, "\n")
            
            leftX, leftY, rightX, rightY = fSplit(X, Y, maxFeature)
            self.buildBranch(leftX, leftY, curr_node.left, depth-1)
            self.buildBranch(rightX, rightY, curr_node.right, depth-1)
        else:
            if item1Count > item2Count:
                curr_node.label = item1Label
                curr_node.data = item1Label
            else:
                curr_node.label = item2Label
                curr_node.data = item2Label
        return 
        
    
    """
    Function:    buildStump
    Description: builds decision stump using a predetermined feature
    Input:       X - matrix holding the features
                 Y - vector holding the target training values
                 featureNum - the number of the feature being used
    Output:      None
    """
    def buildStump(self, X, Y, featureNum):
        self.root = DecisionTreeNode(featureNum)
        self.root.testValue = X[0][featureNum]
        self.root.data = featureNum+1
        leftX, leftY, rightX, rightY = fSplit(X, Y, featureNum)
        
        self.root.left = DecisionTreeNode(None)
        self.root.left.label = fDetermineMajority(leftY)
        self.root.left.data = self.root.left.label
        
        self.root.right = DecisionTreeNode(None)
        self.root.right.label = fDetermineMajority(rightY)
        self.root.right.data = self.root.right.label
    
    
    """
    Function:    readTree
    Description: Inputs data into the tree and returns the expected
                 values after it transverse through the tree
    Input:       testX - matrix holding the features
    Output:      expectedY - vector holding the expected values for each
                             row of features
    """
    def readTree(self, testX):
        expectedY = []
        
        for num in range(len(testX)):
            curr_node = self.root
            
            while curr_node.testNum != None:
                if testX[num][curr_node.testNum] == curr_node.testValue:
                    curr_node = curr_node.right
                else:
                    curr_node = curr_node.left
            expectedY.append(curr_node.label)
            
        return expectedY
    
    """
    Function:    str
    Description: prints depiction of the decision tree
    Input:       None
    Output:      None
    """
    def __str__(self):
        return print_tree(self)
        
    
            
"""
Function:    fSplit
Description: Tests each input against a feature and splits the date set 
             into two, right being where the instances matches with the 
             feature and left being where the instance does not match
Input:       X - matrix holding the features
             Y - vector holding the target training values 
             featNum - the row vector number of the feature being tested
Output:      leftX - matrix where the row does not match the feature
             leftY - corresponding vector where the row does not match 
             the feature
             leftX - matrix where the row matches the feature
             leftY - corresponding vector where row matches the feature
"""
def fSplit(X, Y, featNum):
    leftX = []
    leftY = []
    rightX = []
    rightY = []
        
    testValue = X[0][featNum]
    for num in range(len(X)):
        if X[num][featNum] == testValue:
            rightX.append(X[num])
            rightY.append(Y[num])
        else: 
            leftX.append(X[num])
            leftY.append(Y[num])
                
    return leftX, leftY, rightX, rightY
    


"""
Function:    fDetermineMajority
Description: Determines which class is contained the most in the target
             value vector
Input:       Y - vector holding the target training values 
Output:      majority - the label of the class that is contained in 
                        Y the most 
"""
def fDetermineMajority(Y):
    item1Count, item2Count, item1Label, item2Label = fCount(Y)
    
    if item1Count > item2Count:
        majority = item1Label
    else:
        majority = item2Label
        
    return majority



"""
Function:    fCount
Description: Counts how many of each class is in the target value vector
             and gives the class labels
Input:       Y - vector holding the target training values 
Output:      countItem1 - the number of items of class 1
             countItem2 - the number of items of class 2
             item1Label - the label for class 1
             item2Label - the label for class 2
"""
def fCount(Y):
    countItem1 = 0
    countItem2 = 0
    item1Label = None
    item2Label = None
    
    if len(Y) != 0:
        item1Label = Y[0]
    
    for item in Y:
        if item == item1Label:
            countItem1 += 1
        else:
            countItem2 += 1
            if item2Label == None:
                item2Label = item
            
    return countItem1, countItem2, item1Label, item2Label
    


"""
Function:    fUncertainty
Description: Calculates the uncertainty within a single set of training
             example
Input:       Y - vector holding the target training values 
Output:      Uncertainty - the entropy with said set of training values
"""   
def fUncertainty(Y):
    item1, item2, item1Label, item2Label = fCount(Y)
    total = item1 + item2
    
    item1H = 0
    item2H = 0
    
    if item1 != 0:
        item1H = (item1/total)*math.log(item1/total, 2)
    if item2 != 0:
        item2H = (item2/total)*math.log(item2/total, 2)
    
    return -(item1H + item2H)



"""
Function:    fBenefitOfSplit
Description: Calculutes the gain in information that would occur is splitting
             the data set using a certain feature
Input:       X - matrix holding the features
             Y - vector holding the target training values 
             featNum - the row vector number of the feature being tested 
Output:      Benefit of Split - the reduction of the uncertainity after 
             the split
"""   
def fBenefitOfSplit(X, Y, featNum):
    uncert = fUncertainty(Y)
    
    leftX, leftY, rightX, rightY = fSplit(X, Y, featNum)
    
    remainUncert1 = fUncertainty(leftY)
    prop1 = len(leftY)/len(Y)
    
    remainUncert2 = fUncertainty(rightY)
    prop2 = len(rightY)/len(Y)
    
    return uncert - (prop1 * remainUncert1 + prop2 * remainUncert2)
    
    
    
"""
Function:    fReadFile
Description: Opens a file and appends the first column to a vector and 
             the remaining columns to a matrix
Input:       fileName - the name of the file that will be opened
Output:      X - matrix holding the features
             Y - vector holding the target training values 
"""   
def fReadFile(fileName):
    X = []
    Y = []
    
    file = open(fileName, 'r')
    
    for line in file:
        features = []
        line = line.split(',')
        for num in range(1, len(line)):
            features.append(line[num].strip())
        X.append(features)
        Y.append(line[0])
        
    file.close()
    return X, Y



"""
Function:    fErrorRate
Description: Determines the percantage of incorrect expected values 
             compared to their true values.
Input:       Y - vector holding the true values
             YNew - vector holding the predicted values
Output:      Error rate - percentage of incorrect values
"""   
def fErrorRate(Y, YNew):
    correct = 0
    
    for item in range(len(Y)):
        if Y[item] != YNew[item]:
            correct += 1
    
    return correct/len(Y)


"""
Function:    main
Description: Opens and retieves the data from file, test for 
             information gain for all features, build decision stump
             and decision tree.
Input:       None
Output:      None
"""
decisionStump = DecisionTree() 
decisionTree = DecisionTree()

trainX, trainY = fReadFile("SPECT-train.csv")
testX, testY = fReadFile("SPECT-test.csv")

maxBenefit = 0
maxFeature = None
for feature in range(len(trainX[0])):
    reduction = fBenefitOfSplit(trainX, trainY, feature)

    print("feature ", feature+1, ": ", reduction)
    if reduction > maxBenefit:
        maxBenefit = reduction
        maxFeature = feature

print("\nDecision Stump")    
decisionStump.buildStump(trainX, trainY, maxFeature)

print(decisionStump)

trainYNew = decisionStump.readTree(trainX)
print("Training error rate: ", fErrorRate(trainY, trainYNew))

testYNew = decisionStump.readTree(testX)
print("Testing error rate: ", fErrorRate(testY, testYNew))

print("\nDecision Tree")
decisionTree.buildTree(trainX, trainY, 3)

print(decisionTree)

trainYNew = decisionTree.readTree(trainX)
print("Training error rate: ", fErrorRate(trainY, trainYNew))

testYNew = decisionTree.readTree(testX)
print("Testing error rate: ", fErrorRate(testY, testYNew))
