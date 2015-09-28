Implementation of the Wang(1994) Mixed Logistic Regression Models. The paper uses Binomial responses, but the model can accommodate binary response.


## Data files:

- **heart_scale** from Cj Lin's distributed liblinear
- **tribolium** from Wang(1994)

## Issues:

- In this implementation, I kept using Numpy matrix to save and pass data between different modules. Since Python community recommends using Numpy arrays, I should change the matrix implementation to an array implementation.