# Scalability

Evaluating Scalabilily of the Bayesian Optimization.

In the context of stormwater control, we define scalability as the ability of a control approach to converge onto a viable control strategy as the number of controllable assets increase in a stormwater network.


## Methodology

### Network and Rainfall 

Scalability of the proposed control approach is evaluated on scenario gamma in pystorms library.
In this scenario, a network of 11 controllable stormwater basins are draining into a downstream point.
In each of these basin, we can regulate the outlet valve position to control outflows.
These basins are receiving flows from semi-urban watershed of ~4sq.mi.
This scenario is driven by a 25 year 6 hour design storm.
Please refer to pystorms for more specifics of the water network. 

### Metrics 

There are two specific metrics we wish to use of quantifying the scalability of the proposed control approach.

1. Time till convergence: Given an additional controllable asset, how long does it take for the control approach to converge on to a new solution
2. Time of each iteration: With addition of each new controller asset, does the amount of time for each evaluation change. 
   
These metrics help us to understand the limitations and strengths of the proposed control approach.



### Implementation 


## Results 

1. Parametric Nature of the GP
2. Dependence on Random Seed - Initialization
3. Scaling to Larger Systems 


4. As the number of evaluations increase, given the parametric nature of the approach, simulation time increases. This is not dependent on the number of control points. 
5. Hence as the number of control points increase, we would need more iterations to search though the solution space.

This is only for this approach. 

Time till convergence is specific to the network and rainfall. 

Generalizations we can make are the following.

1. Random seed matters as the number of control points increases 
2. As the number of iterations increases, this dependece reduces.
3. Number of iterations need is proportional to the number of control points 
4. Nothing surprizing there. It is how th method works.

## Discussion 

What does all of this mean.

For simple problems, we can just use the approach. We dont not have to worry about random seed to anything. Should work out the box. 
For larger problems, the number of the control points increases, starting point becomes important. But again if you run them longer you can get away from this dependece.
But rather than running to longer, having a set random seeds and fixed set of iterations to search the solution space would the best apparoach.
As the solutions space increase, we cannot gaurentee the optimality of the solution. But using random stants, we can gauratee that we have achive local optimal

Having said this, this is far more efficient and intrepretable approach than GA. This approach would give you uncertabity quantification and those amazing things.
