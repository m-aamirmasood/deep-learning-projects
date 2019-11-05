#########################################
# kNN: k Nearest Neighbors

# Input:      newInput: vector to compare to existing dataset (1xN)
#             dataSet:  size m data set of known vectors (NxM)
#             labels: 	data set labels (1xM vector)
#             k: 		number of neighbors to use for comparison 

# Output:     the most popular class label
#########################################

from numpy import *
import operator


# create a dataset which contains 4 samples with 2 classes
def createDataSet():
    # create a matrix: each row as a sample
    group = array([[5.7,2.8,4.5,1.3], [6.3,2.5,4.9,1.5], [6.0,2.7,5.1,1.6], [7.0,3.2,4.7,1.4],
                   [6.1,2.9,4.7,1.4], [6.4,2.8,5.6,2.1], [6.4,3.2,5.3,2.3], [5.4,3.0,4.5,1.5],
                   [6.7,3.1,4.7,1.5], [6.0,2.2,4.0,1.0], [5.6,2.5,3.9,1.1], [5.8,2.7,5.1,1.9],
                   [6.7,3.1,4.4,1.4], [6.3,3.3,4.7,1.6], [6.2,2.8,4.8,1.8], [5.8,2.8,5.1,2.4],
                   [6.1,2.8,4.0,1.3], [5.8,2.7,4.1,1.0], [5.5,2.4,3.7,1.0], [6.4,2.8,5.6,2.2],
                   [6.7,3.1,5.6,2.4], [7.2,3.6,6.1,2.5], [7.4,2.8,6.1,1.9], [7.3,2.9,6.3,1.8],
                   [6.5,3.0,5.5,1.8], [5.6,3.0,4.1,1.3], [6.7,3.0,5.2,2.3], [5.8,2.6,4.0,1.2],
                   [5.1,2.5,3.0,1.1], [6.8,3.2,5.9,2.3], [5.5,2.3,4.0,1.3], [6.1,2.6,5.6,1.4],
                   [6.6,3.0,4.4,1.4], [5.0,2.0,3.5,1.0], [5.2,2.7,3.9,1.4], [6.9,3.1,5.4,2.1],
                   [5.6,2.8,4.9,2.0], [6.5,3.0,5.8,2.2], [5.9,3.0,5.1,1.8], [6.0,2.9,4.5,1.5],
                   [5.5,2.6,4.4,1.2], [5.0,2.3,3.3,1.0], [6.4,3.2,4.5,1.5], [4.9,2.5,4.5,1.7],
                   [6.9,3.1,5.1,2.3], [6.1,3.0,4.6,1.4], [5.7,2.5,5.0,2.0], [7.7,2.8,6.7,2.0],
                   [6.5,3.2,5.1,2.0], [6.3,2.7,4.9,1.8], [7.6,3.0,6.6,2.1], [6.5,2.8,4.6,1.5],
                   [7.2,3.0,5.8,1.6], [6.8,3.0,5.5,2.1], [6.1,2.8,4.7,1.2], [5.9,3.0,4.2,1.5],
                   [5.6,3.0,4.5,1.5], [7.7,3.0,6.1,2.3], [6.1,3.0,4.9,1.8], [6.5,3.0,5.2,2.0],
                   [6.7,3.0,5.0,1.7], [6.9,3.2,5.7,2.3], [5.9,3.2,4.8,1.8], [6.4,3.1,5.5,1.8],
                   [5.5,2.5,4.0,1.3], [6.4,2.7,5.3,1.9], [6.7,3.3,5.7,2.1], [5.6,2.7,4.2,1.3],
                   [6.0,2.2,5.0,1.5], [6.3,3.3,6.0,2.5], [5.5,2.4,3.8,1.1], [6.3,2.9,5.6,1.8],
                   [5.7,3.0,4.2,1.2], [7.7,3.8,6.7,2.2], [6.6,2.9,4.6,1.3], [6.3,2.3,4.4,1.3],
                   [7.7,2.6,6.9,2.3], [5.7,2.9,4.2,1.3], [6.3,2.8,5.1,1.5], [6.2,2.2,4.5,1.5]])
    labels = ['0', '0', '0', '0',
              '0', '1', '1', '0',
              '0', '0', '0', '1',
              '0', '0', '1', '1',
              '0', '0', '0', '1',
              '1', '1', '1', '1',
              '1', '0', '1', '0',
              '0', '1', '0', '1',
              '0', '0', '0', '1',
              '1', '1', '1', '0',
              '0', '0', '0', '1',
              '1', '0', '1', '1',
              '1', '1', '1', '0',
              '1', '1', '0', '0',
              '0', '1', '1', '1',
              '0', '1', '0', '1',
              '0', '1', '1', '0',
              '1', '1', '0', '1',
              '0', '1', '0', '0',
              '1', '0', '1', '0']  # four samples and two classes
    return group, labels


# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex