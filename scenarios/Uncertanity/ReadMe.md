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

But we cannot EI as the acquisition function. We need a function for identifying a new sample when there is noise. 
This is where we can use knowledge graidient. 

**Combining KG and Hetroscodastic GP, we can propagate the uncertanity though the simulation engine.**

Though we can use KG, it might be too much for the paper. For now, we can just use expected improvement and be ok with it. 

For the example, we shall consider a synthetic example. Where we have two ponds with syth flows. 

-----------

#### Sampling rainfall distributions for quantifying uncertanity

| Pro | Cons |
| --- | -----|
|When uniformly sampled, we would be weighting the rain event that is most probable and the rain event that is highly unlikely, by a similar magnitude. Hence, we end with a conservative solution. | Complicated and requires a bit of math |

-----------


## Things to do:
1. Reorganize the notes books to be more cohesive.
    - GP implementation 
    - Bayesian Optimization with Noise - Should have modular bayesian optimization with RBF kernel and WhiteNoise kernel 
    - Uncertanity quantification with single basin 
    - Uncertanity quantification with AA network
    - Generate sameples using Brandon uncertanity paper

-----------
There might be two things happening. 

My inital hypothesis was that the randomness would be proportional to the control action. But may be not. 

When you take a action, the response you see is can be anything. Sort of like a stochatic slot machine. We know the priors on the machine. Like the mean and variance of the rainevent. But when we are quantifying it. Does it depend on actions ?

Let me be 100% on it not quantifying uncertanity.
Generate samples and train hetrogp on it see if it is able to get the emperical results.

1. The noise is not dependent on input
2. 

