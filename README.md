# BayOpt
Bayesian Methods for the  calibration and control of storm water networks

# Performance
1. System Scale Control Example - done
# Scalability
1. 11 Basin Example 
	TODO: Re run them to remove errors
	TODO: Plot heat maps
# Uncertanity 
1. Decide on the which path to take. 
   
![](Uncertain_workflow.png)

There is an other way to do it too, but we can explore it later.

For the final thing, I need to have a plot with more uncertanity.
Also most important double check everything to ensure that HETRO GP implementation is right.

The thing is, when we do this, we need to force the GP to take more samples. or we wont get the accurate uncertanity. 
But the GP might not do it. So that is the drawback. 
