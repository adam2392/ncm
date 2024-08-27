import torch

from .scm import truncated_normal


def collider_bar_digit_scm(intervention_idx, labels):
    """Generate parameters for the SCM of the MNIST dataset with a bar following a collider structure.

    The color-bar will be a function of both the digit and the color-digit.

    Parameters
    ----------
    intervention_idx : int
        The intervention index.
    labels : tensor of shape (n_samples,)
        The labels of each of the MNIST digit samples. This corresponds to the
        digit.

    Returns
    -------
    causal_labels : dict
        A dictionary of:
            - digit : tensor of shape (n_samples,)
                The digit label.
            - color_digit : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - color_bar : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
    digit_idx : list of int
        The indices of each image we want in the dataset. This may contain
        replicates of the indices of size (n_samples,).
    """
    n_samples = labels.shape[0]
    causal_labels = dict()

    # Digit is just uniformly distributed more or less in the MNIST dataset
    color_digit = torch.zeros(n_samples)
    color_bar = torch.zeros(n_samples)

    if intervention_idx == 0:
        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit = torch.normal(
            mean=0.5, std=0.2, size=(n_samples, 1)
        ).squeeze()  # 10 possible digits, evenly spaced means from 0 to 1
        digit = torch.Tensor(labels)
        digit_idx = torch.arange(n_samples)

        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 0]] * n_samples)
    elif intervention_idx in (1, 2, 3, 4):
        # Define the skewed probabilities (higher for digits 5-9)
        digit_probabilities = torch.tensor(
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35]
        )
        digit_probabilities[:5] *= intervention_idx * 2
        # Normalize the probabilities to sum to 1 (optional but recommended)
        digit_probabilities /= digit_probabilities.sum()

        # probability of sampling each image
        sampling_probabilities = digit_probabilities[labels]

        # Sample with replacement using the skewed probabilities
        digit_idx = torch.multinomial(sampling_probabilities, n_samples, replacement=True)
        # Get the sampled labels
        digit = labels[digit_idx]

        # Intervention 1: mean equal to 1 - color-digit
        # color_bar = truncated_normal(1 - color_digit, 0.1, 0, 1, n_samples)
        # Intervention 2: skewed distribution towards 0.9
        # color_digit = skewed_normal(intervention_idx / 10., 1.0 / (color_digit + 0.5), n_samples)
        color_digit = truncated_normal(intervention_idx / 5.0, 0.2, 0, 1, n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[1, 1, 0]] * n_samples)
    else:
        raise ValueError("Invalid intervention_idx. Must be 0, 1, 2 or 3.")

    # for each digit, condition on the corresponding color-digit and sample the color-bar
    for i in range(10):
        mask = digit == i
        num_samples = mask.sum().item()
        color_bar[mask] = truncated_normal(color_digit[i], 0.2, 0, 1, num_samples)

    causal_labels.update({"digit": digit, "color_digit": color_digit, "color_bar": color_bar})

    return causal_labels, digit_idx
