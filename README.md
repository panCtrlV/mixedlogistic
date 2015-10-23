Implementation of the Wang(1994) Mixed Logistic Regression Models. The paper uses Binomial responses, but the model can accommodate binary response.


## Data files:

- **heart_scale** from Cj Lin's distributed liblinear

- **tribolium** from Wang(1994)

- Python pickled **MNIST** data, available through [deeplearning.net link](http://deeplearning.net/data/mnist/mnist.pkl.gz)
    
    The pickled file represents a tuple of 3 elements : the training set, the validation set and the testing set. Each of the three elements is a pair-formed tuple containing the images (nImages * nFeatures array) and the corresponding labels (1-d array). An image is represented as numpy 1-dimensional array of 784 (28 x 28) float values between 0 and 1 (0 stands for black, 1 for white). The labels are numbers between 0 and 9 indicating which digit the image represents.

## Issues:

- In this implementation, I kept using Numpy matrix to save and pass data between different modules. Since Python community recommends using Numpy arrays, I should change the matrix implementation to an array implementation.