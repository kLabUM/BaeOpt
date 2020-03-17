# Uncertanity Quanitfucation using Bayesian Optimization and GP

## Things to note:
### Stromwater Example:
        - Stormwater example might not be the best sample. Need to think of a better example.
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

## Log:

In our current approach we assume that data has a uniform noise. 
$$ f(x) = g(x) + N(\mu, \sigma)$$
In the context of stormwater systems, when we are choosing an action $x$, say to close all valves, then the performance we observe is only proportional to the $x$, because rainevents (even though being sampled) are uniform for all the action. If we sample enough times the depedncy on rain will go away. So we can (may be) say that uncertanity of associated with rainfall is dependent on the inputs.
$$ N \propto x$$
So we can use hetrostocastic GP to estimate the uncertanity, which can then be coupled with a acquision function that samples the points multiple times.



**Pro/Con - Using a rain which is input dependent**
Pro:
1. When uniformly sampled, we would be weighting the rain event that is most probable and the rain event that is highly unlikely, by a similar magnitude. Hence, we end with a conservative solution.
    
Cons:
1. Complication 


## Things to do:
1. Reorganize the notes books to be more cohesive.
    - GP implementation 
    - Bayesian Optimization with Noise - Should have modular bayesian optimization with RBF kernel and WhiteNoise kernel 
    - Uncertanity quantification with single basin 
    - Uncertanity quantification with AA network
    - Generate sameples using Brandon uncertanity paper
2. May be use two kernels to get the noise and mean better
3. Figure out hetroscodastic gaussian processes.
