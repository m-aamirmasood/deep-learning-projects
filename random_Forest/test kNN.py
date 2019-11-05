import kNN
from numpy import *

dataSet, labels = kNN.createDataSet()

test = array([[6.2,2.9,4.3,1.3], [6.7,3.3,5.7,2.5], [6.9,3.1,4.9,1.5], [7.1,3.0,5.9,2.1],
             [6.0,3.4,4.5,1.6], [6.8,2.8,4.8,1.4], [5.7,2.6,3.5,1.0], [6.0,3.0,4.8,1.8],
             [5.8,2.7,5.1,1.9], [6.4,2.9,4.3,1.3], [6.3,2.5,5.0,1.9], [7.2,3.2,6.0,1.8],
             [6.7,2.5,5.8,1.8], [6.3,3.4,5.6,2.4], [5.6,2.9,3.6,1.3], [7.9,3.8,6.4,2.0],
             [5.8,2.7,3.9,1.2], [5.7,2.8,4.1,1.3], [6.2,3.4,5.4,2.3], [4.9,2.4,3.3,1.0]])
test_label = array([0, 1, 0, 1,
                    0, 0, 0, 1,
                    1, 0, 1, 1,
                    1, 1, 0, 1,
                    0, 0, 1, 0])
correct = 0
i = 0
while i < 16:
    if test_label[i] == kNN.kNNClassify(test[i], dataSet, labels, 5):
        correct += 1
        print("correct")
    print("\n")
    i += 1
    print(test_label[i])
    print(kNN.kNNClassify(test[i], dataSet, labels, 5))