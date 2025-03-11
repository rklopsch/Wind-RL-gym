import numpy as np
from scipy.stats import norm


def pearson_ci(r: float, n: int, confidence: float = 0.95):
    """
    Computes the confidence interval for a Pearson correlation coefficient.

    Parameters:
        r (float): Pearson correlation coefficient.
        n (int): Sample size.
        confidence (float): Confidence level (default is 0.95).

    Returns:
        tuple: (lower bound, upper bound) of the confidence interval.
    """
    # Fisher transformation
    z = 0.5 * np.log((1 + r) / (1 - r))

    # Standard error
    SE_z = 1 / np.sqrt(n - 3)

    # Critical value for the given confidence level
    z_critical = norm.ppf(1 - (1 - confidence) / 2)

    # Compute confidence interval in z-space
    z_lower = z - z_critical * SE_z
    z_upper = z + z_critical * SE_z

    # Transform back to r-space
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    return float(r_lower), float(r_upper)


if __name__=='__main__':
    print(pearson_ci(0.6, 100))
