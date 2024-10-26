"""Collate function used for irregular and mesh data."""

from typing import List, Tuple, Literal

import torch
from torch.utils.data import default_collate

class Collate:
    """The default collate function used for the dataloader."""
    def __call__(self, batch):
        return default_collate(batch)

class CompositeCollate(Collate):
    """A composite collate function that applies multiple collate functions."""
    def __init__(
        self,
        collate_fns: List[Collate],
    ):
        self.collate_fns = collate_fns
    
    def __call__(self, batch):
        for collate_fn in self.collate_fns:
            batch = collate_fn(batch)
        return batch

class CollateIrregular(Collate):
    """Collate function used for irregular data."""
    
    def __init__(self, strategy: Literal['padding', 'resample', 'nothing'] = 'resample', max_samples: int | None = None):
        self.strategy = strategy
        self.max_samples = max_samples

    def __call__(
        self, batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        """
        Takes a list of tuple of tensors with the following structure:
        (
            ([num_coords, num_samples], [num_samples]), 
            [num_coords, num_samples] | None,
        )
        where num_coords is the number of coordinates in the input data and num_samples is the number of samples in the batch.
        The first tensor is the coordinates, the second tensor is the target and the third tensor is the conditioning.

        It then returns a batch with the following structure:
        (
            ([batch_size, num_coords, max_samples], [batch_size, max_samples]), 
            [batch_size, num_coords, max_samples] | None,
        )
        where batch_size is the number of batches and max_samples is the maximum number of samples in all of the input list.
        """
        all_conds = []
        all_coords = []
        all_samples = []
        max_samples = self.max_samples if self.max_samples is not None else max([len(data[1]) for data, _ in batch])
        for i in range(len(batch)):
            data, cond = batch[i]
            coords, targets = data
            num_samples = len(targets)
            if self.strategy == 'nothing':
                # do nothing
                all_conds.append(cond)
                all_coords.append(coords)
                all_samples.append(targets)
            elif self.strategy == 'padding':
                # padd with resampled elements to match the maximum resolution
                idx = torch.randint(num_samples, (max_samples - num_samples,))
                all_conds.append(torch.cat([cond, cond[idx]]) if cond is not None else None)
                all_coords.append(torch.cat([coords, coords[:, idx]], dim=-1))
                all_samples.append(torch.cat([targets, targets[idx]], dim=-1))
            elif self.strategy == 'resample':
                # sample without replacement
                idx = torch.randint(num_samples, (max_samples,))
                all_conds.append(cond[idx] if cond is not None else None)
                all_coords.append(coords[:, idx])
                all_samples.append(targets[idx])
            else:
                raise ValueError(f"Invalid collate strategy {self.strategy}")
            
        return (
            (torch.stack(all_coords), torch.stack(all_samples)),
            torch.stack([cond for _, _, cond in batch]) if cond is not None else None,
        )

class MakeConditional(Collate):
    """A collate function that adds an additional None tensor to the batch."""
    def __call__(self, batch):
        return [(b, None) for b in batch]
