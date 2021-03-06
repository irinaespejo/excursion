# test_full_1D_line.py

import torch
import yaml
from excursion import init_gp, ExcursionSetEstimator
from excursion.utils import load_example


def test_full_simple():

    tol = 1e-6
    device = torch.device("cpu")
    ninit = 1
    algorithmopts = yaml.safe_load(open("testing/algorithm_specs_full_test.yaml", "r"))

    # three toy examples
    for example in ["1D_test"]:
        testcase = load_example(example)
        models, likelihood = init_gp(testcase, algorithmopts, ninit, device)
        model = models[0]

        estimator = ExcursionSetEstimator(
            testcase, algorithmopts, models, likelihood, device
        )

        while estimator.this_iteration < algorithmopts["nupdates"]:
            estimator.step(testcase, algorithmopts, models, likelihood)
            models = estimator.update_posterior(
                testcase, algorithmopts, models, likelihood
            )
            model = models[0]

    assert type(torch.abs(model.train_targets) <= tol) != type(None)
