from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from gpytorch.means import LinearMean, ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import GreaterThan

def get_kernel_sklearn(ndim, kernel_type):
    if kernel_type == 'const_rbf':
        length_scale = [1.]*ndim
        kernel = ConstantKernel() * RBF(length_scale_bounds=[0.1, 100.0], length_scale=length_scale)
    elif kernel_type == 'tworbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2, 100]) + \
                 ConstantKernel() * RBF(length_scale_bounds=[100., 1000.0]) + \
                 WhiteKernel(noise_level_bounds=[1e-7, 1e-4])
    elif kernel_type == 'onerbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2, 100]) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1])
    else:
        raise RuntimeError('unknown kernel')
    return kernel
    
    
    
def get_kernel_gpytorch(kernel_type):
    if kernel_type == "RBF":
        covar_module = ScaleKernel(RBFKernel(
                    lengthscale_constraint=GreaterThan(lower_bound=0.1)
                )
            )
    else:
        raise NotImplementedError()
            
    return covar_module
    
    
def get_mean_gpytorch(mean_type:str, train_x):
    if mean_type == "Linear":
            ndim = train_x.shape[1]
            mean_module = LinearMean(ndim=ndim)  
    elif mean_type == "Constant":
            mean_module = ConstantMean()
    elif mean_type == "Zero":
            mean_module = ZeroMean()
    else:
        raise NotImplementedError()
    
    return mean_module