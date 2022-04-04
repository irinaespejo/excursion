import gpytorch
from .base import ExcursionModel
from .utils import get_kernel_gpytorch, get_mean_gpytorch
import numpy as np
from torch.optim import Adam
from torch import from_numpy


class ExactGP(ExcursionModel, gpytorch.models.ExactGP):
    """This is a guassian process used to compute the excursion model. Most if not all excursion models will be a
    gaussian process.
    """
    def __init__(self, train_x, train_y, likelihood:str, mean:str, kernel:str, fit_optimizer:str):
        self.likelihood = likelihood
        if(likelihood == 'GaussianLikelihood'):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        else:
            raise NotImplementedError('Only gaussan likelihoods implemented')
            
        # cast to torch
        train_x = from_numpy(train_x)
        train_y = from_numpy(train_y)

        # init gpytorch gp
        super(ExactGP, self).__init__(train_x, train_y, likelihood) 
        self.mean_module = get_mean_gpytorch(mean, train_x)
        self.covar_module = get_kernel_gpytorch(kernel)
        
       

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        

    def update_model(self, x, y):
        """
        Updates training data (does not re-fit model hyper-parameters).
        """

        inputs = x
        targets = y if x.shape[1] != 1 else y.flatten()

        if not hasattr(self, "X_train_") and not hasattr(self, "y_train_"):  # No data; add it
            self.X_train_ = inputs
            self.y_train_ = targets.ravel()

        else:
            self.X_train_ = np.concatenate((self.X_train_, inputs), axis=0)

            y_train_ = np.concatenate((self.y_train_, targets), axis=0)
            self.y_train_ = y_train_.ravel()

        # # # Should actually check to see if the data is correct, TO DO
        # self.set_train_data(inputs=inputs, targets=targets, strict=False)


    def fit_model(self, fit_optimizer):
        if not hasattr(self, "X_train_") and not hasattr(self, "y_train_"):  # No data; add it
            raise ValueError("The model's does not have any training data to fit with.")
        
        else:
            training_iter = 100
            X_train = self.train_inputs[0]
            y_train = self.train_targets

            if fit_optimizer == "Adam":
                optimizer = Adam(
                    [{"params": self.parameters()},],  # Includes GaussianLikelihood parameters
                    lr=0.1,
                    eps=10e-6,
                )

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

                for i in range(training_iter):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from gp
                    output = self.forward(X_train)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_train)
                    loss.sum().backward(retain_graph=True)
                    optimizer.step()

            else:
                raise NotImplementedError('Fitting the model with optimizer other than Adam is not implemented')
                    
