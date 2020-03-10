# Uncertanity Quanitfucation using Bayesian Optimization and GP

## Things to note:

### Gaussian Processes:
#### Hyperparameters
        - They are sensitive to the choice of the hyperparameters. Hence, the quality of the GP depends on the optimization approach used for estimating these hyperparameters.
        - ML is often used as the objective function for estimating these hyperparameters.
#### Kernels:
        - Inverting the kernel is the most involved process. Care must be taken to ensure that the kernel matrix is PD. Choosing a PD kernel should handle that issue. But I am inclined to think that the issue is the numerical. Numpy matrix inversion might be causing issues when the size of the matrix is big. I should try cholecky decomposition. But it my previous attemps, I've noticed that it wasnt any better. May be I am not applying it right. I need to double check what is happening here. 
        - In the mean time, I can use the Gpy.

### Estimated Noise:
        - Noise estimated by the Bayesian Optimization is not exactly the same as that of the uncertanity in the objective function. 
        - This may be due to how samples are generated, and the optimization approach being used to search the hyperparameter space. 
        - Increasing the sample count will help us estimate the uncertanity. But as the solution space increases, we cannot sample all the space to learn the exact uncertanity. Hence, GP might be a resonable approach to learn the how the solution space looks like.
        
## Things to do:
1. Reorganize the notes books to be more cohesive.
    - GP implementation 
    - Bayesian Optimization with Noise - Should have modular bayesian optimization with RBF kernel and WhiteNoise kernel 
    - Uncertanity quantification with single basin 
    - Uncertanity quantification with AA network
    - Generate sameples using Brandon uncertanity paper