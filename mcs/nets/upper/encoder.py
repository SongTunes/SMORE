import torch
import torch.nn as nn

from mcs.nets.layers import MultiHeadAttention, SelfAttention


class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.init_W = nn.Linear(2 * embed_dim, embed_dim)

        self.attention = SelfAttention(
            MultiHeadAttention(n_heads=1, embed_dim=embed_dim, need_W=True)
        )

    def forward(self, x, mask=None):
        """

        @param x: [bs, max_n_couriers, 2*embed_dim]
        @param mask:
        @return:
        """
        x = self.init_W(x)  # [bs, max_nc, embed_dim]
        x = self.attention(x, mask=mask)
        x = torch.mean(x, dim=1, keepdim=True)
        return x  # [bs, 1, embed_dim]


def generate_state_hidden(state, node_embeddings):
    """

    @param state:
        state.static_h: [batch, max_n_couriers, embed_dim]
        state.dynamic: [bs, t, max_n_couriers]
    @param node_embeddings: [bs, n_sensingtasks, embed_dim]
    @return:
    """

    batch_size, state_len, max_n_couriers = state.dynamic.size()
    _, n_sensingtasks1, embed_dim = node_embeddings.size()

    state_h0 = torch.gather(
        node_embeddings[:, :, None, :].expand(batch_size, n_sensingtasks1, max_n_couriers, embed_dim),
        dim=1,
        index=state.dynamic[..., None].clone().contiguous().expand(batch_size, state_len, max_n_couriers, embed_dim)
    )  # [bs, t, max_n_couriers, embed_dim]
    state_h0 = torch.mean(state_h0, dim=1)  # [bs, max_n_couriers, embed_dim]
    # state_h0 = torch.sum(state_h0, dim=1)  # [bs, max_n_couriers, embed_dim]
    state_h0 = torch.cat([state.static_h, state_h0], dim=2)  # [bs, max_n_couriers, 2*embed_dim]

    return state_h0  # [bs, max_n_couriers, 2*embed_dim]


if __name__ == '__main__':
    pass
