Machine learning methods for the task of classification. For this purpose, four classifiers named as logistic regression, neural networks, random forest and SVM were implemented. Multiclass logistic regression was implemented on MNIST training dataset and then it was tested on MNIST testing dataset and USPS testing dataset. The same was done for neural network, random forest and SVM. The confusion matrix of each classifier was observed to oversee the relative strengths and weaknesses of each classifier. Finally, the results of individual classifiers were combined with majority voting to measure the overall combined performance.

# Datasets #
## MNIST dataset ##
28x28 grayscale handwritten digit images and identify them as digits among 0,1,2,3,4,5,6,7,8,9.
## USPS dataset ##
16*16 grayscale pixels for digits.

# Implementation #
## Logistic Regression ##
Logistic regression model is as below:
𝒚(𝒙, 𝒘) = 𝒔𝒐𝒇𝒕𝒎𝒂𝒙(𝒘𝚻𝝓(𝒙))
w is the weight vector in this equation and 𝜙 is the input matrix. The softmax activation gave the probability distribution of input over the aforementioned 10 labels. There were 10 labels and in order to classify the data properly, one-hot vector notation was used where class Ck is a 10-dimensional vector of class labels. In that notation, class C2 can be defined as:
C3= (0,1,0,0,0,0,0,0,0,0)T
The MNIST training data consisted of dimensions 50,000x28x28 or 5000x784 and the testing data was of dimensions 50000x28x28 or 50000x784 and the testing data
was of the dimension 10,000x784. The weights were initially assigned to zero. The loss function used was negative log-likelihood.

![alt text](loss.png)

where gradient was given by:

![alt text](gradient_descent.png)

The gradient descent criterion was used to minimize the log likelihood function and weights were updated accordingly. Best results were achieved when the learning rate was set to 0.01 and the 5000 iterations were done over the data. Lambda of 10 was used for regularization term.
## Neural Network ##
Keras was used for the neural network for classification task. Reshape and to_categorical methods were used to reorganize the data into desired format and to transform the labels into one hot vector notation. The model defined for the neural network was densely connected neural network with one hidden layer where each input is connected to each output by a weight. The model was a simple sequential neural network with a single possible output for each input which is enough for this
classification problem. The input layer consisted of 32 nodes. The labels were categorized into 10 values in a list which contained the correct labels of each number for the input layer. Sigmoid activation function was used for that layer. SGD was used as an optimizer. The model was then trained on mini-batches of size
128 and over 100 epochs. The output layer consisted of 10 nodes and softmax was used for that purpose. Finally, accuracy was calculated on both training and testing datasets.
## Random Forest ##
Random forest was implemented using the sklearn packages. Random forest classifier defined in sklearn was used for the purpose of implementing the required algorithm. The parameter n_estimator was defined as 10, since we had 10 labels. The .fit method was used to fit the data on the classifier and .score method was used to measure the accuracy of the classifier.
## SVM ##
Support vector machine was also implemented using the packages defined in sklearn. SVC package was used for that purpose and the kernel ‘rbf’ was chosen. The kernel helps to map the data into a higher dimensional space. The accuracy was calculated.
## Ensemble Classifier ##
The predictions of all the classifiers were then combined to create an ensemble classifier and majority voting was used to determine the label of the image.
## No Free Lunch Theorem ##
Hume pointed out that ‘even after the observation of the frequent or constant conjunction of objects, we have no reason to draw any inference concerning any object beyond those of which we have had experience’. No free lunch theorem states that if an algorithm does well on one problem then it necessarily pays for that in other problems. When the above-mentioned classifiers were trained on MNIST dataset, they performed well on MNIST testing data but did not fare too well on USPS dataset. This is shown in Table 1:

| classifier          | MNIST | USPS |
|---------------------|:-------:|------:|
| Logistic Regression | 74%   | 34%  |
| Neural Network      | 93%   | 41%  |
| Random Forest       | 89%   | 39%  |
| SVM                 | 91%   | 39%  |
| Ensemble            | 91%   | 40%  |
