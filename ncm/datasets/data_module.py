from pathlib import Path
from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Sampler, random_split

from .colorbar import CausalBarMNIST
from .digitcolorbar import CausalDigitBarMNIST


# Custom Stratified Sampler
class StratifiedSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(labels)
        self.unique_labels = np.unique(labels)
        self.label_indices = {label: np.where(labels == label)[0] for label in self.unique_labels}
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        num_per_class = self.batch_size // len(self.unique_labels)

        for _ in range(self.num_samples // self.batch_size):
            batch_indices = []
            for label in self.unique_labels:
                label_indices = np.random.choice(
                    self.label_indices[label], num_per_class, replace=False
                )
                batch_indices.extend(label_indices)

            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)

        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


class MultiDistrDataModule(LightningDataModule):
    """
    Data module for multi-distributional data.

    Attributes
    ----------
    medgp: MultiEnvDGP
        Multi-environment data generating process.
    num_samples_per_env: int
        Number of samples per environment.
    batch_size: int
        Batch size.
    num_workers: int
        Number of workers for the data loaders.
    intervention_targets_per_distr: Tensor, shape (num_envs, num_causal_variables)
        Intervention targets per environment, with 1 indicating that the variable is intervened on.
    log_dir: Optional[Path]
        Directory to save summary statistics and plots to. Default: None.
    intervention_target_misspec: bool
        Whether to misspecify the intervention targets. If true, the intervention targets are permuted.
        I.e. the model received the wrong intervention targets. Default: False.
    intervention_target_perm: Optional[list[int]]
        Permutation of the intervention targets. If None, a random permutation is used. Only used if
        intervention_target_misspec is True. Default: None.
    flatten: bool
        Whether to flatten the data. Default: False.

    Methods
    -------
    setup(stage=None) -> None
        Setup the data module. This is where the data is sampled.
    train_dataloader() -> DataLoader
        Return the training data loader.
    val_dataloader() -> DataLoader
        Return the validation data loader.
    test_dataloader() -> DataLoader
        Return the test data loader.
    """

    def __init__(
        self,
        root,
        graph_type,
        batch_size: int,
        stratify_distrs: bool = True,
        label: int = 0,
        num_workers: int = -1,
        train_size: float = 0.9,
        val_size: float = 0.05,
        transform=None,
        log_dir: Optional[Path] = None,
        dataset_name: str = None,
        subsample=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_dir = Path(log_dir) if log_dir is not None else log_dir
        self.dataset_name = dataset_name

        self.stratify_distrs = stratify_distrs
        self.transform = transform
        self.train_size = train_size
        self.val_size = val_size

        self.root = root
        self.label = label
        self.graph_type = graph_type
        self.subsample = subsample

        if self.dataset_name not in ["digitcolorbar", "colorbar"]:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_name == "digitcolorbar":
            self.dataset = CausalDigitBarMNIST(
                root=self.root,
                graph_type=self.graph_type,
                train=True,
                n_jobs=None,
                transform=self.transform,
                subsample=self.subsample,
            )
        elif self.dataset_name == "colorbar":
            self.dataset = CausalBarMNIST(
                root=self.root,
                graph_type=self.graph_type,
                train=True,
                n_jobs=None,
                transform=self.transform,
            )

        train_size = int(self.train_size * len(self.dataset))
        val_size = int(self.val_size * (len(self.dataset) - train_size))
        test_size = len(self.dataset) - train_size - val_size

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_split(self.dataset, [train_size, val_size, test_size])

        if self.stratify_distrs:
            distr_labels = [x[1][-1] for x in self.train_dataset]
            self.train_sampler = StratifiedSampler(distr_labels, self.batch_size)

            distr_labels = [x[1][-1] for x in self.val_dataset]
            self.val_sampler = StratifiedSampler(distr_labels, self.batch_size)
        else:
            self.train_sampler = None
            self.val_sampler = None

    @property
    def meta_label_strs(self):
        return self.dataset.meta_label_strs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
