import torch
import torch.nn as nn

from mcs.nets.layers import MultiHeadAttention, DotProductAttention
# from mcs.nets.upper.utils import Env
from mcs.nets.Sampler import TopKSampler, CategoricalSampler

MAX_NUM = 99999


class DecoderCell(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, clip=10., **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.max_n_couriers = 45

        self.Wks1 = nn.Linear(embed_dim + embed_dim, embed_dim, bias=False)
        self.Wvs = nn.Linear(embed_dim + embed_dim, embed_dim, bias=False)
        self.Wks2 = nn.Linear(embed_dim + embed_dim, embed_dim, bias=False)

        self.Wkd1 = nn.Linear(1, embed_dim, bias=False)
        self.Wvd = nn.Linear(1, embed_dim, bias=False)
        self.Wkd2 = nn.Linear(1, embed_dim, bias=False)

        self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_step = nn.Linear(embed_dim + 32, embed_dim, bias=False)

        self.MHA = MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, need_W=False)
        self.SHA = DotProductAttention(clip=clip, return_logits=True, head_depth=embed_dim)
        # SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
        # self.env = Env

    def compute_attention(self, node_embeddings, step_context, mask):
        Ks1 = self.Wks1(node_embeddings)
        Vs = self.Wvs(node_embeddings)
        Ks2 = self.Wks2(node_embeddings)

        K1 = Ks1
        K2 = Ks2
        V = Vs

        Q1 = self.Wq_step(step_context)
        Q2 = self.MHA([Q1, K1, V], mask=mask)
        Q2 = self.Wout(Q2)
        logits = self.SHA([Q2, K2, None], mask=mask)
        return logits.squeeze(dim=1)

    def forward(self, mask, step_context, state_h0, encoder_output, decode_type='sampling'):
        """

        @param mask:
        @param step_context:
        @param state_h0: [bs, 45, 2*embed_dim]
        @param encoder_output:
        @param decode_type:
        @return:
        """

        node_embeddings, graph_embedding, customers_embedding = encoder_output
        batch_size, _, _ = node_embeddings.size()

        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
        log_ps, tours = [], []

        logits = self.compute_attention(state_h0, step_context, mask)

        p = torch.softmax(logits / 8, dim=-1)
        log_p = torch.log_softmax(logits / 8, dim=-1)

        next_node = selecter(log_p)  # [bs, 1]

        # print(next_node)
        # print(torch.gather(p, 1, next_node))

        tours.append(next_node.squeeze(1))
        log_ps.append(log_p)

        tours = torch.stack(tours, 1)  # [bs, tour_len=1]
        log_ps = torch.stack(log_ps, 1)  # [bs, tour_len=1, max_n_couriers]

        return tours, log_ps


if __name__ == '__main__':
    pass
