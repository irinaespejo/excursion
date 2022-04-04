from .base import AcquisitionFunction
from .utils import entropy
import numpy as np
import os


class GPMES(AcquisitionFunction):
    def __init__(self, batch=False):
        self.batch = batch
        self.acq_vals = None

    def acquire(self, gp, thresholds, X_pointsgrid):
        """
        Calculates information gain of choosing x_candidate as next point to evaluate.
        Performs this calculation with the Maximum Entropy Search approximation. Roughly,
        MES(x_candidate) =  H[Y(x_candidate)|S(x=j)] 
        Notation: MES(x_candidate) =  H[Y(x_candidate)|S(x=j)] 

        """

        self.acq_vals = self._acquire(gp, thresholds, X_pointsgrid)

        X_train = gp.X_train_.tolist()
        for i, cacq in enumerate(X_pointsgrid[np.argsort(self.acq_vals)]):
            if cacq.tolist() not in X_train:
                newx = cacq
                return newx

    def _acquire(self, gp, thresholds, X_pointsgrid):
        try:
            from joblib import Parallel, delayed
            nparallel = int(os.environ.get('EXCURSION_NPARALLEL', os.cpu_count()))
            result = Parallel(nparallel)(
                delayed(entropy)(x_candidate, gp, thresholds, X_pointsgrid) for x_candidate in X_pointsgrid)
            return np.asarray(result)
        except ImportError:
            return np.array([entropy(x_candidate, gp, thresholds, X_pointsgrid) for x_candidate in X_pointsgrid])
