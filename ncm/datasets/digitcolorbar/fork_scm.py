import torch

from .scm import skewed_normal, truncated_normal


def collider_bar_digit_scm(intervention_idx, labels):
    """Generate parameters for the SCM of the MNIST dataset with a bar following a collider structure.

    The digit will cause the color_digit and color_bar.

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
    """
    n_samples = labels.shape[0]
    causal_labels = dict()

    # Digit is just uniformly distributed more or less in the MNIST dataset
    digit = torch.Tensor(labels)
    color_digit = torch.zeros(n_samples)
    color_bar = torch.zeros(n_samples)

    if intervention_idx == 0:
        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit_means = torch.linspace(
            0, 1, 10
        )  # 10 possible digits, evenly spaced means from 0 to 1
        color_digit_stds = 0.15 * torch.ones(10)  # Standard deviation of 0.15 for each digit

        # Generate color_bar based on the intervention index
        color_bar_means = torch.linspace(0, 1, 10)[
            ::-1
        ]  # 10 possible digits, evenly spaced means from 0 to 1
        color_bar_stds = 0.15 * torch.ones(10)  # Standard deviation of 0.15 for each digit

        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 0]] * n_samples)
    elif intervention_idx == 1:
        # Intervention 1: mean equal to 1 - color-digit
        # color_bar = truncated_normal(1 - color_digit, 0.1, 0, 1, n_samples)
        # Intervention 2: skewed distribution towards 0.9
        color_bar = skewed_normal(0.1, 1.0 / (color_digit + 0.5), n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 1]] * n_samples)
    elif intervention_idx == 2:
        # Intervention 2: skewed distribution towards 0.9
        color_bar = skewed_normal(0.9, 1.0 / (color_digit + 0.5), n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 1]] * n_samples)
    elif intervention_idx == 3:
        # Intervention 3: hard normal distribution on color_digit
        n_samples = labels.shape[0]
        color_digit = truncated_normal(0.5, 0.2, 0, 1, n_samples)
        # color_bar = truncated_normal(0.5, 0.2 / (color_digit + 0.5), 0, 1, n_samples)
        color_bar = truncated_normal(1.0 / (color_digit + 1), 0.1, 0, 1, n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 1, 0]] * n_samples)
    else:
        raise ValueError("Invalid intervention_idx. Must be 0, 1, 2 or 3.")

    # sample the color-digit conditioned on the digit
    for i in range(10):
        mask = digit == i
        num_samples = mask.sum().item()
        # color_digit[mask] = torch.distributions.Normal(
        #     color_digit_means[i], color_digit_stds[i]
        # ).sample((num_samples,))
        color_digit[mask] = truncated_normal(
            color_digit_means[i], color_digit_stds[i], 0, 1, num_samples
        )

    for i in range(10):
        mask = digit == i
        num_samples = mask.sum().item()
        color_bar[mask] = truncated_normal(color_bar_means[i], color_bar_stds[i], 0, 1, num_samples)

    causal_labels.update({"digit": digit, "color_digit": color_digit, "color_bar": color_bar})

    return causal_labels
