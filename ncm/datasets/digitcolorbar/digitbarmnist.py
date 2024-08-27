from pathlib import Path

import torch
from torch.utils.data import Dataset


# Define the dataset loader for digit dataset
class CausalDigitBarMNIST(Dataset):
    def __init__(
        self,
        root,
        graph_type,
        train=True,
        transform=None,
        target_transform=None,
        n_jobs=None,
        subsample=None,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.n_jobs = n_jobs
        self.graph_type = graph_type

        root = Path(root)

        # load data from disc
        self.data = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-imgs-train.pt"
        )
        self.labels = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-labels-train.pt"
        )
        if isinstance(self.labels, list):
            self.labels = torch.vstack(self.labels)

        self.intervention_targets = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-targets-train.pt"
        )
        if isinstance(self.intervention_targets, list):
            self.intervention_targets = torch.vstack(self.intervention_targets)

        if not all(
            [
                len(self.data) == len(self.labels),
                len(self.data) == len(self.intervention_targets),
            ]
        ):
            raise ValueError("Data, labels and intervention targets must have the same length.")

        if subsample is not None:
            self.data = self.data[:subsample]
            self.labels = self.labels[:subsample]
            self.intervention_targets = self.intervention_targets[:subsample]

    @property
    def intervention_targets_per_distr(self):
        return [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a sample from the image dataset.

        The target composes of the meta-labeling:
        - width
        - color
        - fracture_thickness
        - fracture_num_fractures
        - label
        """
        img, meta_label, target = (
            self.data[index],
            self.labels[index],
            self.intervention_targets[index],
        )

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)

        return img, meta_label, target

    @property
    def meta_label_strs(self):
        return ["digit", "color_digit", "color_bar", "distr_idx"]

    @property
    def digit_idx(self):
        return 0

    @property
    def color_digit_idx(self):
        return 1

    @property
    def color_bar_idx(self):
        return 2

    @property
    def digit(self):
        return self.labels[:, 0]

    @property
    def color_digit(self):
        return self.labels[:, 1]

    @property
    def color_bar(self):
        return self.labels[:, 2]

    @property
    def latent_dim(self):
        return self.labels.shape[1]

    @property
    def distribution_idx(self):
        return self.labels[:, 3]
