import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import gpytorch 
import excursion
import time
import simplejson
from excursion.models import ExactGP_RBF, GridGPRegression_RBF
from excursion.active_learning import acq
from excursion.utils import get_first_max_index
import excursion.plotting.onedim as plots_1D
import excursion.plotting.twodim as plots_2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def init_gp(testcase, algorithmopts, ninit, device):
    likelihood_type = algorithmopts['likelihood']['type']
    modelopts = algorithmopts['model']['type']
    kernelopts = algorithmopts['model']['kernel']
    n_dims = testcase.n_dims
    epsilon = float(algorithmopts['likelihood']['epsilon'])
    dtype = torch.float64

    #
    # TRAIN DATA
    #
    X_grid = torch.Tensor(testcase.X_plot).to(device, dtype)
    init_type = algorithmopts['init_type']
    noise_dist= MultivariateNormal(torch.zeros(ninit), torch.eye(ninit))

    if(init_type=='random'):
        indexs = np.random.choice(range(len(X_grid)), size = ninit, replace=False)
        X_init = X_grid[indexs].to(device, dtype)
        noises = epsilon*noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
    elif(init_type=='worstcase'):
        X_init = [X_grid[0]]
        X_init = torch.Tensor(X_init).to(device, dtype)
        noises = epsilon*noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
    elif(init_type=='custom'):
        raise NotImplementedError('Not implemented yet')
    else:
        raise RuntimeError('No init data specification found')  

    #
    #LIKELIHOOD
    #
    if(likelihood_type=='GaussianLikelihood'):
        if(epsilon > 0.):
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=torch.tensor([epsilon])).to(device, dtype)
        elif(epsilon == 0.):
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.tensor([epsilon])).to(device, dtype) 

    else:
        raise RuntimeError('unknown likelihood')

    #
    #GAUSSIAN PROCESS
    #
    if(modelopts == 'ExactGP' and kernelopts =='RBF'):
        model = ExactGP_RBF(X_init, y_init, likelihood).to(device, dtype)
    elif(modelopts == 'GridGP' and kernelopts =='RBF'):
        grid_bounds = [(testcase.rangedef[0][0], testcase.rangedef[0][1]), (testcase.rangedef[1][0], testcase.rangedef[1][1])]
        grid_size = testcase.rangedef[0][2]*testcase.rangedef[1][2]
        grid = torch.zeros(int(grid_size), len(grid_bounds), dtype=torch.double)
        for i in range(len(grid_bounds)):
            grid[:, i] = torch.linspace(grid_bounds[i][0] , grid_bounds[i][1], int(grid_size), dtype=torch.double)

        model = GridGPRegression_RBF(grid, X_init, y_init, likelihood).to(device, dtype)
    else:
        raise RuntimeError('unknown gpytorch model')

    #fit
    print('X_init ', X_init)
    print('y_init ', y_init)
    model.train()
    likelihood.train()
    excursion.fit_hyperparams(model, likelihood)

    return model, likelihood



def get_gp(X, y, likelihood, algorithmopts, testcase, device):
    modelopts = algorithmopts['model']['type']
    kernelopts = algorithmopts['model']['kernel']
    #
    #GAUSSIAN PROCESS
    #

    #to
    X = X.to(device)
    y = y.to(device)

    if(modelopts == 'ExactGP' and kernelopts =='RBF'):
        model = ExactGP_RBF(X, y, likelihood).to(device)
    elif(modelopts == 'GridGP' and kernelopts =='RBF'):
        grid_bounds = [(testcase.rangedef[0][0], testcase.rangedef[0][1]), (testcase.rangedef[1][0], testcase.rangedef[1][1])]
        grid_size = testcase.rangedef[0][2]*testcase.rangedef[1][2]
        grid = torch.zeros(int(grid_size), len(grid_bounds), dtype=torch.double)
        for i in range(len(grid_bounds)):
            grid[:, i] = torch.linspace(grid_bounds[i][0] , grid_bounds[i][1], int(grid_size), dtype=torch.double)

        model = GridGPRegression_RBF(grid, X, y, likelihood).to(device)
        
    else:
        raise RuntimeError('unknown gpytorch model')

    #fit
    model.train()
    likelihood.train()
    excursion.fit_hyperparams(model, likelihood)

    return model


def fit_hyperparams(gp, likelihood, optimizer: str='Adam'):
    training_iter = 100
    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    if(optimizer=='LBFGS'):
        optimizer = torch.optim.LBFGS([
            {'params': gp.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1, line_search_fn=None)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        def closure():
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f outputscale: %.3f  noise: %.3f' % (
            #i + 1, training_iter, loss.item(),
            #gp.covar_module.base_kernel.lengthscale.item(),
            #gp.covar_module. outputscale.item(),
            #gp.likelihood.noise.item()
            #))
            return loss


    if(optimizer=='Adam'):

        optimizer = torch.optim.Adam([
            {'params': gp.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1, eps=10e-6)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.sum().backward()
            optimizer.step()

            #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f outputscale: %.3f  noise: %.3f' % (
            #i + 1, training_iter, loss.item(),
            #gp.covar_module.base_kernel.lengthscale.item(),
            #gp.covar_module. outputscale.item(),
            #gp.likelihood.noise.item()
            #))




class ExcursionSetEstimator():

    def __init__(self, testcase, algorithmopts, model, likelihood, device):
        self.x_new = torch.zeros(1, testcase.n_dims,  dtype=torch.float64)
        self.y_new = torch.zeros(1,1, dtype=torch.float64)
        self.acq_values = []

        self.this_iteration = 0
        self.confusion_matrix = []
        self.pct_correct = []
        self.walltime_step = []
        self.walltime_posterior = []
        self.device = device
        self.dtype = torch.float64

        self._acq_type = algorithmopts['acq']['acq_type'] 
        self._X_grid = testcase.X
        self._epsilon = algorithmopts['likelihood']['epsilon']
        self._n_dims = testcase.n_dims


    def get_diagnostics(self, testcase, model, likelihood):
        thresholds = [-np.inf] + testcase.thresholds.tolist() + [np.inf]
        X_eval = (testcase.X)
        noise_dist = MultivariateNormal(torch.zeros(len(X_eval)), torch.eye(len(X_eval)))
        noise = self._epsilon*noise_dist.sample(torch.Size([])).to(self.device, self.dtype)
        y_true = testcase.true_functions[0](X_eval).to(self.device, self.dtype) + noise

        model.eval()
        likelihood.eval()
        y_pred = likelihood(model(X_eval.to(self.device, self.dtype))).mean

        def label(y):
            for j in range(len(thresholds)-1):
                if(y<thresholds[j+1] and y>thresholds[j]):
                    return 'c_'+str(j) 

        labels_pred = [label(y) for y in y_pred]
        labels_true = [label(y) for y in y_true]

        #force y_true = y_train for those x in dataset
        
        conf_matrix = confusion_matrix(labels_true, labels_pred)
        self.confusion_matrix.append(conf_matrix)
        pct = np.diag(conf_matrix).sum()*1./len(X_eval)
        self.pct_correct.append(pct)
        return None


    def step(self, testcase, algorithmopts, model, likelihood):
        #track wall time
        start_time = time.process_time()
        self.this_iteration += 1
        print('Iteration ', self.this_iteration)
       
        #order grid indexs by maxmium acquitision function value
        new_indexs, self.acq_values = self.get_new_indexs(model, testcase)

        #discard those points already in dataset
        new_index = get_first_max_index(model, new_indexs, testcase)

        #get x, y
        self.x_new = testcase.X[new_index].reshape(1,-1)
        self.x_new = self.x_new.to(self.device, self.dtype)
        noise_dist = MultivariateNormal(torch.zeros(1), torch.eye(1))
        noise = self._epsilon*noise_dist.sample(torch.Size([])).to(self.device, self.dtype)
        self.y_new = testcase.true_functions[0](self.x_new).to(self.device, self.dtype) + noise
        self.y_new = self.y_new.to(self.device, self.dtype)
        #update training data
        #model = self.update_posterior(testcase, algorithmopts, model, likelihood)
        
        #track wall time
        end_time = time.process_time() - start_time
        self.walltime_step.append(end_time)
        return self.x_new, self.y_new


    def get_new_indexs(self, model, testcase):
        acquisition_values_grid = []
        for x in self._X_grid:
            #print('****x ', x.shape, type(x), x)
            value = acq(model,testcase, x.view(1,-1).to(self.device, self.dtype), self._acq_type, self.device, self.dtype)
            acquisition_values_grid.append(value)
            new_indexs = np.argsort(acquisition_values_grid)[::-1]
        return new_indexs, acquisition_values_grid


    def update_posterior(self, testcase, algorithmopts, model, likelihood):
        #track wall time
        start_time = time.process_time()
        if(self._n_dims==1):
            inputs_i = torch.cat((model.train_inputs[0], self.x_new),0).flatten()
            targets_i = torch.cat((model.train_targets.view(-1,1), self.y_new),0).flatten()
        else:
            inputs_i = torch.cat((model.train_inputs[0], self.x_new),0)
            targets_i = torch.cat((model.train_targets, self.y_new),0).flatten()

        model.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)
        model = get_gp(inputs_i, targets_i, likelihood, algorithmopts, testcase, self.device)

        likelihood.train()
        model.train()
        fit_hyperparams(model, likelihood)

        #track wall time
        end_time = time.process_time() - start_time
        self.walltime_posterior.append(end_time)

        return model


    def plot_status(self, testcase, model, acq_values, outputfolder):
        if(self._n_dims==1):
            plots_1D.plot_GP(model, testcase, \
                            acq=self.acq_values, \
                            acq_type=self._acq_type, \
                            x_new=self.x_new, \
                            device=self.device, \
                            dtype=self.dtype) 
            plt.savefig(outputfolder+'/'+str(self._n_dims)+'D_'+str(self.this_iteration)+'_'+str(self._acq_type)+'.png')
        
        elif(self._n_dims==2):
            fig = plt.figure()
            plot = plots_2D.plot_GP(plt, model, testcase, self.device, self.dtype) 
            plt.tight_layout()
            figname = outputfolder+str(self._n_dims)+'D_'+str(self.this_iteration)+ \
                      '_'+str(self._acq_type)+'.png'
            plt.savefig(figname)

        else:
            pass


    def print_results(self, outputfolder, testcase, algorithmopts):
        print('Printing results...')

        #plot pct_correct
        filename_pct = outputfolder+'/pct_correct_'+self._acq_type+'_'+algorithmopts['init_type']+'.txt'
        filename_pct_img = outputfolder+'/pct_correct_'+self._acq_type+'_'+algorithmopts['init_type']+'.png'
        plt.clf()
        plt.title('percentage correct classification')
        plt.plot(range(self.this_iteration), self.pct_correct, label=self._acq_type)
        plt.xlabel('iteration')
        plt.ylabel('%')
        plt.legend()
        plt.hlines(y=1, xmax=self.this_iteration, xmin=0, color='grey', linestyle='--')
        plt.savefig(filename_pct_img)

        tick_marks = np.arange(len(testcase.thresholds)+2)
        c = ['c_'+str(j) for j in range(len(testcase.thresholds)+1)]

        #print pct to file
        with open(outputfolder+'pct_correct.txt','w') as f:
            simplejson.dump(self.pct_correct, f)

        # confusion matrix plot
        for i in range(self.this_iteration):
            plt.clf()
            plt.title('Confusion matrix iter='+str(i)+' '+self._acq_type)
            plt.xticks(tick_marks, c, rotation=45)
            plt.yticks(tick_marks, c)
            plt.imshow(self.confusion_matrix[i], cmap='binary')
            for i1 in range(self.confusion_matrix[i].shape[0]):
                for i2 in range(self.confusion_matrix[i].shape[1]):
                    plt.text(i1,i2,self.confusion_matrix[i][i1][i2], ha='center', va='center',color='red')
            plt.tight_layout()
            plt.savefig(outputfolder+'/CF_'+str(i)+'.png')

        #print to file walltimes
        filename_walltime_step = outputfolder + 'walltime_step.txt'
        filename_walltime_posterior = outputfolder + 'walltime_posterior.txt'
        with open(filename_walltime_step,'w') as g:
            simplejson.dump(self.walltime_step, g)
        with open(filename_walltime_posterior,'w') as h:
            simplejson.dump(self.walltime_posterior, h)


        return None
