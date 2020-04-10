import numpy as np
from pyswmm_lite import environment
import baestorm
import GPy
import abc
from six import with_metaclass
import copy


class BOModel(with_metaclass(abc.ABCMeta, object)):
    """
    The abstract Model for Bayesian Optimization
    """

    MCMC_sampler = False
    analytical_gradient_prediction = False

    @abc.abstractmethod
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        "Augment the dataset of the model"
        return

    @abc.abstractmethod
    def predict(self, X):
        "Get the predicted mean and std at X."
        return

    # We keep this one optional
    def predict_withGradients(self, X):
        "Get the gradients of the predicted mean and variance at X."
        return

    @abc.abstractmethod
    def get_fmin(self):
        "Get the minimum of the current model."
        return


# Generate Gaussian Flows
def GaussianSignal(x, amplitude, timetopeak, dispersion):
    flows = (
        amplitude
        * (1.0 / (dispersion * (np.sqrt(2.0 * np.pi))))
        * np.exp(-0.5 * ((x - timetopeak) / dispersion) ** 2)
    )
    return flows


# Objective function
def ObjectiveFunction(actions):
    # Sample the rainevent
    padding_length = 500
    temp_x = np.linspace(-10.0, 10.0, 100)

    def flows_amp(amp):
        return np.pad(
            GaussianSignal(temp_x, amp, -2.0, 3.0), (0, padding_length), "constant"
        )

    # Pick a random stormevent - Uniformly Sampled
    amplitude = np.random.choice(np.linspace(5.0, 10.0, 10), size=1)
    flows = flows_amp(amplitude)
    env = environment(baestorm.load_networks("parallel"), False)

    reward = 0.0
    for time in range(0, len(flows)):
        # Set the gata position
        env._setValvePosition("1", actions[0])
        env._setValvePosition("2", actions[1])

        # Set inflows
        env.sim._model.setNodeInflow("P1", flows[time])
        env.sim._model.setNodeInflow("P2", flows[time])

        # compute performance
        flow = env.methods["flow"]("8")
        if flow > 0.50:
            reward += 10.0 * (flow - 0.50)
        else:
            reward += 0.0

        if (
            env.sim._model.getNodeResult("P1", 4)
            + env.sim._model.getNodeResult("P2", 4)
            > 0.0
        ):
            reward += 10 ** 5
        else:
            reward += 0.0

        # Record data
        _ = env.step()
    env._terminate()
    return reward


# Setup the Gaussain Process for quantifying the error
class GPModel_Hetro(BOModel):
    """
    Class for handling hetroscodastic noise.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available.
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """

    analytical_gradient_prediction = (
        True
    )  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(
        self,
        kernel=None,
        noise_var=None,
        exact_feval=False,
        optimizer="bfgs",
        max_iters=1000,
        optimize_restarts=5,
        sparse=False,
        num_inducing=10,
        verbose=True,
        ARD=False,
    ):
        # Kernel has to incude hetroscodaticc gaussian noise
        # could be included as sum.
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD

        self.count = 0

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(
                self.input_dim, variance=1.0, ARD=self.ARD
            )  # + GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel

        # --- define model
        noise_var = Y.var() * 0.01 if self.noise_var is None else self.noise_var

        if not self.sparse:
            self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        else:
            self.model = GPy.models.SparseGPRegression(
                X, Y, kernel=kern, num_inducing=self.num_inducing
            )

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(
                1e-9, 1e6, warning=False
            )  # constrain_positive(warning=False)

    def updateModel(
        self, X_all, Y_all, X_new, Y_new, samples=10, iterations=100, epsilon=0.01
    ):
        """
        Updates the model with new observations.
        """
        print("Iteration :", self.count)
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.kern = GPy.kern.RBF(input_dim=1) + GPy.kern.White(input_dim=1)
            self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts == 1:
                self.model.optimize(
                    optimizer=self.optimizer,
                    max_iters=self.max_iters,
                    messages=False,
                    ipython_notebook=False,
                )
            else:
                self.model.optimize_restarts(
                    num_restarts=self.optimize_restarts,
                    optimizer=self.optimizer,
                    max_iters=self.max_iters,
                    verbose=self.verbose,
                )

        # Create Kernels for 2 and 3 GP
        # How does a multi-diemtional white noise kernel look?
        kernel2 = GPy.kern.RBF(input_dim=1) + GPy.kern.White(input_dim=1)
        kernel3 = GPy.kern.RBF(input_dim=1) + GPy.kern.WhiteHeteroscedastic(
            input_dim=1, num_data=X_all.shape[0]
        )

        eps = np.inf
        count = 0
        var_pre = np.zeros(X_all.shape[0])
        while True:
            if count > iterations:
                break
            # Get the means and varainces for all the paramenters
            m, v = self.model.predict(X_all)
            z = np.zeros(X_all.shape[0])  # This might bite me in the back
            for j in range(0, X_all.shape[0]):
                var = 0.0
                for i in range(0, samples):
                    var += 0.5 * (Y_all[j] - np.random.normal(m[j], v[j])) ** 2
                z[j] = var / samples
            z = np.log(z)
            # Step 2:
            gp2 = GPy.models.GPRegression(X_all, z.reshape(-1, 1), kernel2)
            gp2.optimize(
                optimizer=self.optimizer,
                max_iters=self.max_iters,
                messages=False,
                ipython_notebook=False,
            )
            gp2.optimize_restarts(num_restarts=10)
            # gp2.optimize_restarts(num_restarts=5)
            m_n, v_n = gp2.predict(X_all)
            # Step-3:
            gp3 = GPy.models.GPRegression(X_all, Y_all.reshape(-1, 1), kernel3)
            gp3.optimize(
                optimizer=self.optimizer,
                max_iters=self.max_iters,
                messages=False,
                ipython_notebook=False,
            )
            gp3.optimize_restarts(num_restarts=10)
            gp3.kern.parts[1].variance = np.exp(m_n).reshape(X_all.shape[0])

            m, v = gp3.predict(X_all)

            diff = var_pre - v
            eps = np.dot(diff.T, diff)[0][0]
            self.model = copy.deepcopy(gp3)
            var_pre = copy.deepcopy(v)
            count += 1
            print("epsilon:", eps)
            if eps < epsilon:
                break

            print("counter:", count)

        self.count += 1

    def _predict(self, X, full_cov, include_likelihood):
        if X.ndim == 1:
            X = X[None, :]
        m, v = self.model.predict(
            X, full_cov=full_cov, include_likelihood=include_likelihood
        )
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict(self, X, with_noise=True):
        """
        Predictions with the model.
        Returns posterior means and standard deviations at X.
        Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, False, with_noise)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_covariance(self, X, with_noise=True):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        _, v = self._predict(X, True, with_noise)
        return v

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        return self.model.predict(self.model.X)[0].min()

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim == 1:
            X = X[None, :]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:, :, 0]
        dsdx = dvdx / (2 * np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.model.posterior_covariance_between_points(x1, x2)
