The purpose of this project was to differentiate between two images that whether a word is written by the same person or not. For this purpose, three techniques named as linear regression, logistical regression and neural networks were used. For linear regression, noise and basis objective functions were defined and regularization was used to contain the overfitting problem. The closed-form solution and gradient solution were found and the root mean square error was calculated for the models. The same was done for logistic regression and then a neural network model was generated to solve the same problem. Finally, the hyperparameters were changed to different settings and the accuracy of different models was compared.

## Linear Regression Model ##
Linear regression function for the model was:
y(x,w)=w^T ϕ(x)
w is weight vector in and ϕ is the basis function. The first element of w, w0 is the bias in this case. The function of the basis function is to map vectors to a scalar value. The basis function used for this model was radial basis function with the equation:
ϕ_j (x)= e^((-1/2 〖(x-μ_j)〗^T Σ_j^(-1) (x-μ_j)))
μ_j is the center of basis function and Σ_j^(-1)defines the spatial spread.

## Regularization ##
Regularization was added to the error function to reduce the problem of overfitting and for this purpose a weight decay regularizer was used. The equation is as follows:
E(w)=E_D (w)+ λE_w (w)
The equation for weight decay regularizer 
E_w (w)=1/2 w^T w
The goal was to minimize E(w) by selecting appropriate value of w. 

## Closed-form Solution ##
The closed form solution with a regularizer can be found by using the following equation:
w^*=(λI+ϕ^T ϕ)^(-1) ϕ^T t

## Gradient Descent Soltuion ##
Stochastic gradient descent takes the derivative of a weight vector and multiplies it with a learning rate. The learning rate can be changed to manage the convergence time of the model. The new weights are then updated in accordance with the following equation:
w^((τ+1) )=w^τ+Δw^τ
And then we have:
∇E_D=-(t_n-〖w^τ〗^T ϕ(x_n ))ϕ(x_n )
∇E_w=w^τ

## Evaluation ##
The solutions were then evaluated on the basis of root mean square (rms) error, which is defined as:
E_RMS=√(2&(2E(w^*))/N_v )

## Code ##
NumPy was used for the implantation of the problem and k-mean algorithm was also imported from sklearns module to get the k-means clustering. The dataset was read and was divided in training, testing and validation tests. Training data consisted of 80% of the dataset while testing and validation each received 10%. The closed form solution was then calculated. It consisted of finding the Moore-Penrose pseudo inverse matrix. In the first step, K-means algorithms was used to convert the data into different clusters and the mean of centroids was calculated. This vector of means was then used, along with the training data to create big sigma.
The gaussian radial basis function was then calculated by taking the inverse of big sigma and multiplying it with the difference of data matrix and the means vector calculated from the k-means algorithm. This factor was again multiplied with the transpose of the difference. The weights were then calculated and a regularization term was added to prevent the overfitting of data. Similar, process was done for testing and validation sets. The root mean square error was then calculated to measure the performance of the model.
In the same fashion, the weights were initialized for the case of stochastic gradient descent and the derivative was calculated and then multiplied with a fixed learning rate. This term was then added to the previous weight matrix to get the new weights. The error function was then calculated and the root mean square error was also determined to measure the performance.

## Results ##
The hyperparameters of the model were changed for both solutions and the value of root mean square was calculated to see which value gave better results. For closed form solution, clusters of 10, 50, 100 and 200 were chosen. As a general observation, the ERMS value decreased on testing data with increase in number of clusters as shown in table 1 but the trade-off was made in time required for training the data. For increased number of clusters, model took more time to train. The value of lambda didn’t affect the ERMS value dramatically but it usually tended to increase with increase in lambda as shown in table 2. For stochastic gradient descent, the hyperparameters of learning rate, clusters and lambda were changed. As with the case in closed-form solution, the ERMS value decreased for increase in number of clusters as shown in table 3. The lowest value for ERMS was achieved when learning rate was changed to 0.05 as is apparent in table 4. The ERMS value increased for both lower and higher values of learning rate.  Table 5 shows the value of ERMS on different lambda values.

## Tables ##

### Closed-form solution ###

| Clusters | ERMS training | ERMS validation | ERMS testing |
|----------|:---------------:|:-----------------:|:--------------:|
| 10       | 0.549         | 0.538           | 0.627        |
| 50       | 0.54          | 0.537           | 0.619        |
| 100      | 0.536         | 0.530           | 0.613        |
| 200      | 0.529         | 0.528           | 0.60         |

| Lambda | ERMS training | ERMS validation | ERMS testing |
|--------|:---------------:|:-----------------:|:--------------:|
| 0      | 0.549         | 0.538           | 0.628        |
| 0.03   | 0.549         | 0.538           | 0.627        |
| 0.5    | 0.5           | 0.54            | 0.63         |
| 0.9    | 0.6           | 0.59            | 0.69         |

### Stochastic Gradient Descent ###

| Clusters | ERMS training | ERMS validation | ERMS testing |
|----------|:---------------:|:-----------------:|:--------------:|
| 10       | 0.549         | 0.538           | 0.627        |
| 50       | 0.533          | 0.53           | 0.62        |
| 100      | 0.531         | 0.537           | 0.6155        |
| 200      | 0.529         | 0.530           | 0.608         |

### Results ###
Different combinations for the hyperparameters were tried and the best possible root mean square measure was achieved when the clusters were increased to 50 and for closed-form solution. For gradient descent method, rms could be further decreased if the cluster of 100 and learning rate of 0.05 was chosen. The root mean square error value did not change much when other hyperparameters were changed.
