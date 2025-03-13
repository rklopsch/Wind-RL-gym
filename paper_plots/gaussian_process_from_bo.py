import pickle
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound


if __name__ == '__main__':
    with open('../outputs/bo_logs/BO_output.pkl', 'rb') as f:
        bo_logs = pickle.load(f)


    train_x = bo_logs['train_x']
    train_y = bo_logs['train_y']
    lower_bound = -40.
    bounds_diff = 80.

    scaled_x = (train_x - lower_bound) / bounds_diff
    model = SingleTaskGP(scaled_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    predictions = model.posterior(scaled_x).mean
    L1_error = torch.mean(torch.abs(predictions - train_y))
    print(f"Relative L1 error {100 * L1_error / torch.mean(train_y):.2f}%")



