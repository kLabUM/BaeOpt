# Performance

Performance of the Bayesian Optimization Controller is evaluated on scenario gamma. 


## ```gamma-1-basin```

In this scenario, controller maintains the flows in a single basin below a threshold. 

## ```gamma-4-basin```

In this scenario, controller maintains the flows in the 4 basins in series below a desired threshold. 

### Additional Information

Both these scenarios use generic summation functions as their reward function. 

1. In this context, you can see that the functions are convex, hence the solution identified by the controller is globally optimal. 
2. How do you evaluate the convexity of a numerical function.

