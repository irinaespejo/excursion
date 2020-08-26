# test_initialize_gp.py

import torch
import yaml
from excursion import init_gp
from excursion.commandline import load_example


def test_init_gp():

    device = torch.device("cpu")
    n_init = 1
    algorithmopts = yaml.safe_load(open("./algorithm_specs.yaml", "r"))

    # three toy examples
    for example in ["1Dtoyanalysis", "2Dtoyanalysis", "3Dtoyanalysis"]:
        testcase = load_example(example)
        gp = init_gp(testcase, algorithmopts, ninit, device)

        assert type(gp) == torch.distributions.MultivariateNormal