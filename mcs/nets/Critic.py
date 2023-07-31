import torch.nn as nn
import torch.nn.functional as F

from mcs.nets.encoder import GraphAttentionEncoder


class StateCritic(nn.Module):
    """
    Estimate the baseline value.
    """
    def __init__(self, embed_dim=128, n_encode_layers=3, n_heads=8, tanh_clipping=10., FF_hidden=512):
        super(StateCritic, self).__init__()

        self.encoder = GraphAttentionEncoder(embed_dim, n_heads, n_encode_layers, FF_hidden)

        # Define the encoder & decoder models
        self.fc1 = nn.Linear(embed_dim*45, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        x[0] -- depot_data: (batch, 1, 4) --> embed_depot_xy: (batch, embed_dim)
        x[1] -- grids_data: (batch, (grid_size)*n_timeslots), 4)
        x[2] -- customers_data: (batch, max_n_couriers, 2 * max_n_customers)
        x[3] -- tsps_data: (batch, max_n_couriers, 1)
        x[4] -- ntask_data:
        """

        # Use the probabilities of visiting each
        _, _, hidden = self.encoder(x)
        bs, max_n_couriers, embed_dim = hidden.size()
        hidden = hidden.reshape(bs, max_n_couriers*embed_dim)
        # bs, max_n_couriers, embed_dim = x[2].size()

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output
