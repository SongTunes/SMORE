import torch
import torch.nn as nn

from mcs.nets.upper.decoder import DecoderCell
from mcs.nets.upper.encoder import AttentionEncoder, generate_state_hidden
from mcs.nets.functions import get_log_likelihood


class AttentionModel(nn.Module):

    def __init__(self, embed_dim=128, n_heads=8, tanh_clipping=10.):
        super().__init__()

        self.embed_dim = embed_dim
        self.Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping)
        self.max_n_couriers = 45

        self.encoder = AttentionEncoder()
        self.linear_budget = nn.Linear(1, 32)

    def forward(self, x, encoder_output, rest_budget, state, mask_has_no_task, decode_type='greedy'):
        """

        @param x:
            x[0] -- depot_data: (batch, 1, 4) --> embed_depot_xy: (batch, embed_dim)
            x[1] -- grids_data: (batch, (grid_size)*n_timeslots-1), 4)
            x[2] -- customers_data: (batch, max_n_couriers, 2 * max_n_customers)
        @param encoder_output:
        @param rest_budget:
        @param state:
        @param mask_has_no_task:
        @param decode_type:
        @return:
        """

        #
        mask_workerpad = ~(x[2].clone().sum(2, keepdims=True).bool())  # [bs, max_n_couriers, 1]
        mask = mask_workerpad + mask_has_no_task
        # print(mask_has_no_task)
        node_embeddings, graph_embedding, customers_embedding = encoder_output
        # [bs, n_sensingtasks+1, embed_dim]  [bs, embed_dim]  [bs, max_n_couriers, embed_dim]
        state.static_h = customers_embedding
        batch_size, _, _ = node_embeddings.size()

        #
        total_tours = []
        total_logps = []

        #
        state_h0 = generate_state_hidden(state, node_embeddings)  # [bs, max_nc, 2*embed_dim]
        # state_h0 = self.env.generate_state_hidden_transformer(state, depot_grids, node_embeddings, self.encoder)  # [bs, 45, 2*embed_dim]
        # step_context = state_h0.reshape(batch_size, self.max_n_couriers*(self.embed_dim+self.embed_dim)).unsqueeze(1)  # [bs, 1, 45*2*embed_dim]
        #
        state_context = self.encoder(state_h0, mask=mask_workerpad)  # [bs, 1, embed_dim]
        h_budget = self.linear_budget((0.01 * rest_budget).unsqueeze(1).unsqueeze(2))
        step_context = torch.cat([state_context, h_budget], dim=2)  # [bs, 1, embed_dim+32]

        #
        decoder_output = \
            self.Decoder(mask, step_context, state_h0, encoder_output, decode_type=decode_type)

        tours, logps = decoder_output  # [8, 1] [8, 1, 45]

        total_tours.append(tours)
        total_logps.append(logps)

        total_tours = torch.cat(total_tours, dim=1)  # [bs, 1]
        total_logps = torch.cat(total_logps, dim=1)  # [bs, 1, max_n_couriers]

        ll = get_log_likelihood(total_logps, total_tours)  # [bs]
        return ll, total_tours, state_context


if __name__ == '__main__':
    pass
