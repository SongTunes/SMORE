import torch
import torch.nn as nn

from mcs.nets.layers import MultiHeadAttention, SelfAttention


class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.max_n = 100

        self.attention = SelfAttention(
                MultiHeadAttention(n_heads=1, embed_dim=embed_dim, need_W=True)
            )

    def forward(self, x, mask=None):
        """

        @param x: [bs, t, 1, embed_dim]
        @param mask:
        @return:
        """
        bs, t, one, embed_dim = x.size()
        x = x.transpose(1, 2)  # [bs, 1, t, embed_dim]

        x_ext = torch.zeros(bs, self.max_n, embed_dim, device=self.device)
        x_ext[:, 0:t, :] = x[:, 0, :, :]  # [bs, 40, embed_dim]

        # mask: [bs, 40, 1] bool
        mask = (torch.sum(x_ext, dim=2, keepdim=True) == 0)

        x_ext = self.attention(x_ext, mask=mask)
        x_ext = torch.mean(x_ext, dim=1, keepdim=False)
        # x_ext: [bs, embed_dim]
        return x_ext


def generate_state_hidden_transformer(state, depot_grids, node_embeddings, encoder):
    """
    Generate the Context.
    For Sensing Task Selecting, the Context is:
    ((Delivery Tasks of this Courier), (Sensing Tasks which already selected)).

    state.static: [bs, 1, embed_dim]
    state.dynamic: [bs, t, 1]
    node_embeddings: [bs, n_sensingtasks+1, embed_dim]
    """
    batch_size, state_len, one = state.dynamic.size()
    _, n_nodes, embed_dim = node_embeddings.size()

    state_h0 = torch.gather(
        node_embeddings[:, :, None, :].expand(batch_size, n_nodes, one, embed_dim),
        dim=1,
        index=torch.clamp(state.dynamic - 1, min=0)[..., None].clone().contiguous().expand(batch_size, state_len,
                                                                                           one, embed_dim)
    )  # [bs, t, one, embed_dim]

    state_h0 = encoder(state_h0)  # [bs, embed_dim]
    state_h0 = torch.cat([state.static_h.squeeze(1), state_h0], dim=1)  # [bs, 2*embed_dim]

    return state_h0.unsqueeze(1)  # [bs, 1, 2*embed_dim]

def generate_state_hidden(state, node_embeddings):
    """
    Generate the Context.
    For Sensing Task Selecting, the Context is:
    ((Delivery Tasks of this Courier), (Sensing Tasks which already selected)).

    state.static: [bs, 1, embed_dim]
    state.dynamic: [bs, t, 1]
    node_embeddings: [bs, n_sensingtasks+1, embed_dim]
    """
    batch_size, state_len, one = state.dynamic.size()
    _, n_nodes, embed_dim = node_embeddings.size()

    # code optimization
    # state_h0 == states_h
    state_h0 = torch.gather(
        node_embeddings[:, :, None, :].expand(batch_size, n_nodes, one, embed_dim),
        dim=1,
        index=torch.clamp(state.dynamic-1, min=0)[..., None].clone().contiguous().expand(batch_size, state_len, one, embed_dim)
    )  # [bs, t, one, embed_dim]
    state_h0 = torch.mean(state_h0, dim=1)  # [bs, one, embed_dim]
    # state_h0 = torch.sum(state_h0, dim=1)  # [bs, one, embed_dim]
    state_h0 = state_h0.reshape(batch_size, one * embed_dim)  # [bs, embed_dim]
    state_h0 = torch.cat([state.static_h.squeeze(1), state_h0], dim=1)  # [bs, 2*embed_dim]

    return state_h0.unsqueeze(1)  # [bs, 1, 2*embed_dim]


if __name__ == '__main__':
    pass
