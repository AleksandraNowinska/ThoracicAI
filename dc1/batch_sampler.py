import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler, RandomSampler
from image_dataset import ImageDataset
from typing import Generator, Tuple


class BatchSampler(Sampler):
    """
    Implements an iterable which, given a dataset and batch_size, 
    produces batches of data of the given size. The batches are returned as tuples (images, labels).

    Now uses WeightedRandomSampler for balanced batches while preserving randomness.
    """

    def __init__(self, batch_size: int, dataset: ImageDataset, balanced: bool = False) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.balanced = balanced

        if self.balanced:
            # Compute class weights dynamically based on dataset imbalance
            class_counts = np.bincount(self.dataset.targets)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[self.dataset.targets]

            # Create a Weighted Random Sampler
            self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(self.dataset), replacement=True)
        else:
            # Use default random sampler for unbalanced datasets
            self.sampler = RandomSampler(self.dataset)

    def __len__(self) -> int:
        return (len(self.dataset) // self.batch_size) + 1

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        indices = list(self.sampler)

        # Iterate over dataset in batch_size steps
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Extract images and labels for the batch
            X_batch = [self.dataset[idx][0] for idx in batch_indices]
            Y_batch = [self.dataset[idx][1] for idx in batch_indices]
            
            # Return stacked tensors
            yield torch.stack(X_batch).float(), torch.tensor(Y_batch).long()
