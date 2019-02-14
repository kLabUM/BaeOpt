import torch
import numpy as np
import gpytorch 
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("whitegrid")

# Sample identification
from torch.distributions import constraints, transform_to
import torch.autograd as autograd
import torch.optim as optim
torch.manual_seed(42)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        # Constant Mean 
        self.mean_module = gpytorch.means.ConstantMean()
        # RBF Kenel Scaled 
        wn_variances = torch.randn(train_x.shape)
        self.covr_module = gpytorch.kernels.AdditiveKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        # Computes a forward pass and returns a multivariate normal distribution 
        mean_x = self.mean_module(x)
        covr_x = self.covr_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covr_x)

def EI(x, obs_mean, model, likelihood, xi=0.01):
    """
    Computes the expected inprovement based on the GP
    """
    # Compute the mean and covariance of new data
    observed_new = likelihood(model(x))
    mean_new = observed_new.mean
    vari_new = observed_new.variance
    # Standard Normal Distribution 
    nrml = torch.distributions.normal.Normal(0.0, 1.0)
    # Identify the max from the sampled data
    max_mean = obs_mean.max()
    # Compute Z 
    Z = (mean_new - max_mean - xi)/vari_new
    # Estimate the ei
    ei = (mean_new - max_mean - xi)*nrml.cdf(Z) + vari_new*torch.exp(nrml.log_prob(Z))
    if bool((vari_new > 0.0).numpy()):
        return ei 
    else:
        return ei * 0.0
    
def candidate_id(x_init,
                obs_mean,
                model,
                likelihood,
                lower_bound=0, upper_bound=1):
    """
    Identify the next point to sample based on the acquision function 
    and some pytorch magical beans 
    """
    # transform x to an unconstrained domain
    constraint = constraints.interval(lower_bound, upper_bound)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
    minimizer = optim.LBFGS([unconstrained_x])

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y = -EI(x, obs_mean, model, likelihood)
        autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    return x.detach(), EI(x, obs_mean, model, likelihood).detach()


# Create the function f
def f(X, noise):
    return torch.sin(np.pi*X) - X**2 + 0.7*X + noise * torch.randn(X.shape)

X_sampled = torch.tensor([0.1]) # Some inital starting point 
Y_sampled = f(X_sampled, noise=0.5)

# Set up the number of iter 
num_ties = 100

# Datalog
datalog = {}
datalog["x_next"] = []
datalog["acqui"] = []
datalog["acqui_com"] = []

# Bayopt loop 
for i in range(0, num_ties):
    # Set up the optimization thingy 
    gp_likihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model    = GPModel(X_sampled, Y_sampled, gp_likihood)


    # Train the model based on the X_sampled and Y_sampled 
    
    # Find optimal model hyperparameters
    gp_model.train()
    gp_likihood.train()
    
    ## Optimizer 
    optimizer = torch.optim.Adam([{'params': gp_model.parameters()},], lr=0.1)
    
    ## "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_likihood, gp_model)

    
    ## Tune the hyperparms 
    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = gp_model(X_sampled)
        # Calc loss and backprop gradients
        loss = -mll(output, Y_sampled)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
        
    # Get into evaluation (predictive posterior) mode
    gp_model.eval()
    gp_likihood.eval()
    
    temp_x = []
    temp_a = []
    # Compute the mean of sampled points 
    observed_sam = gp_model(X_sampled)
    mean_sam = observed_sam.mean.detach()

    for i in range(0, 5):
        # Get the candidate for eval 
        x0 = torch.FloatTensor(1).uniform_(0.0, 2.0)
        x_new, acq = candidate_id(x0,
                                  mean_sam,
                                  gp_model,
                                  gp_likihood,
                                  lower_bound=0.0, upper_bound=2.0)
        temp_x.append(x_new)
        temp_a.append(acq)
    
    argmin = torch.max(torch.cat(temp_a), dim=0)[1].item()
    x_new = temp_x[argmin]
    acq = temp_a[argmin]
    datalog["x_next"].append(x_new.numpy())
    datalog["acqui"].append(acq.numpy())
    temp_c = []
    ta = torch.linspace(0.0, 2.0, 10)
    for i in ta:
        temp_c.append(EI(i.reshape(1,), mean_sam, gp_model, gp_likihood))
    datalog["acqui_com"].append(temp_c)

    # Append to the sampled point 
    X_sampled = torch.cat((X_sampled, x_new), dim=0)
    Y_sampled = torch.cat((Y_sampled, f(x_new, noise=0.50)), dim=0)


# Get into evaluation (predictive posterior) mode
gp_model.eval()
gp_likihood.eval()

# Test points are regularly spaced along [0,1]
test_gp_x = torch.linspace(0, 2, 100)

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = gp_likihood(gp_model(test_gp_x))

with torch.no_grad():
    # Initialize plot
    fi, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(X_sampled.numpy(), Y_sampled.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_gp_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_gp_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

plt.show()
