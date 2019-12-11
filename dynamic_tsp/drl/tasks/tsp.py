"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        # self.dataset = torch.rand((num_samples, 2, size))
        # self.dynamic = torch.zeros(num_samples, 1, size)

        original_location = torch.rand((int(num_samples), 2, size))
        original_mask = torch.zeros(int(num_samples), 1, size)
        self.static = original_location
        self.dynamic = torch.cat((original_mask, original_location), 1)

        self.num_nodes = size
        self.size = num_samples
        self.mean = 0.0
        self.std = 0.02


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], [])


    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (x, y) coordinates."""

        # Update the dynamic elements
        # print('The dynamic is:', dynamic)
        # print('The chosen_idx is:', chosen_idx)
        visit = chosen_idx.ne(0)
        # print('The visit is:', visit)

        # Clone the dynamic variable so we don't mess up graph
        dynamic_copy = dynamic.clone()

        # Across the minibatch
        if visit.any():
            # visit_idx = chosen_idx.item()
            # print('visit_idx is:', visit_idx)
            for i in range(len(dynamic_copy)):
                dynamic_copy[i][0][chosen_idx] = 1

            for i in range(len(dynamic_copy)):
                non_visit_set = [j for j in range(self.num_nodes) if dynamic_copy[i][0][j] == 0]
                for idx in non_visit_set:
                    dynamic_copy[i][1][idx] += torch.normal(self.mean, self.std, (1, 1)).squeeze()
                    dynamic_copy[i][2][idx] += torch.normal(self.mean, self.std, (1, 1)).squeeze()

        return torch.tensor(dynamic_copy.data, device=dynamic.device)



def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(dynamic, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    final_coordinates = dynamic[:, 1:, :]
    idx = tour_indices.unsqueeze(1).expand_as(final_coordinates)
    tour = torch.gather(final_coordinates.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()



def render(dynamic, tour_indices, save_path):
    """Plots the found tours."""

    static = dynamic[:, 1:, :]
    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)


def render_static(static, tour_indices, save_path):
    """Plots the found tours."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
