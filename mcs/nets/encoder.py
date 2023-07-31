import torch
import torch.nn as nn

from config import arg_parser
from mcs.nets.layers import EncoderLayer


class GraphAttentionEncoder(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, n_layers=3, FF_hidden=512):
        super().__init__()
        cfg = arg_parser()
        self.return_depot = cfg.return_depot
        self.n_row = cfg.n_rows
        self.n_col = cfg.n_cols

        self.init_W_depot = torch.nn.Linear(2, embed_dim, bias=True)
        self.init_W = torch.nn.Linear(4, embed_dim, bias=True)
        self.max_n_customers = 15
        # self.init_W_customers = torch.nn.Linear(2 * self.max_n_customers, embed_dim, bias=True)
        self.conv = nn.Conv2d(1, 3, kernel_size=2, stride=1)  # [bs, 1, nr, nc] -> [bs, 3, nr-1, nc-1]
        self.linear = nn.Linear(3 * (self.n_row - 1) * (self.n_col - 1), embed_dim)

        # 10*12 graph

        self.encoder_layers_task = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])
        self.encoder_layers_courier = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])

    def forward(self, x, mask2=None):
        """

        @param x:
            x[0] -- depot_data: (batch, 1, 4) --> embed_depot_xy: (batch, embed_dim)
            x[1] -- grids_data: (batch, (grid_size)*n_timeslots-1), 4)  [161, 1757, 4]
            x[2] -- customers_data: (batch, max_n_couriers, 2 * max_n_customers)
            x[3] -- customers_graph: (batch, max_n_couriers, 1, nr, nc)
            x[4] -- ntask_data:
        @param mask2: mask the worker pad
        @return:
            --> concated_customer_feature: (batch, n_nodes-1, 3) --> embed_customer_feature: (batch, n_nodes-1, embed_dim)
            embed_x(batch, n_nodes, embed_dim)

            return: (node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
                =((batch, n_nodes, embed_dim), (batch, embed_dim))
        """

        #
        bs, max_n_couriers, _ = x[2].size()
        if self.return_depot:
            x1 = torch.cat([self.init_W_depot(x[0][:, :, 0:2]), self.init_W(x[1])], dim=1)
        else:
            x1 = torch.cat([self.init_W_depot(x[0].mean(1)[:, None, 0:2]), self.init_W(x[1])], dim=1)
        # x2 = self.init_W_customers(x[2])
        x2_list = []
        for i in range(max_n_couriers):
            x2 = self.conv(x[3][:, i, :, :, :])
            x2 = self.linear(x2.reshape(bs, -1))
            x2_list.append(x2.unsqueeze(1))  # [bs, 1, embed_dim]
        x2 = torch.cat(x2_list, dim=1)  # [bs, max_n_couriers, embed_dim]

        for layer in self.encoder_layers_task:
            x1 = layer(x1)
        for layer in self.encoder_layers_courier:
            x2 = layer(x2, mask=mask2)
        # x1: [bs, grid_size*n_timeslots, embed_dim]
        # x1_mean: [bs, embed_dim]

        return x1, torch.mean(x1, dim=1), x2  # node_embedding and graph embedding and customers_embedding


if __name__ == '__main__':
    pass
