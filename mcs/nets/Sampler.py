import torch
import torch.nn as nn


class Sampler(nn.Module):
    """ args; logits: (batch, n_nodes)
        return; next_node: (batch, 1)
        TopKSampler <=> greedy; sample one with biggest probability
        CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
    """

    def __init__(self, n_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples


class TopKSampler(Sampler):
    def forward(self, logits):
        return torch.topk(logits, self.n_samples, dim=1)[1]  # == torch.argmax(log_p, dim = 1).unsqueeze(-1)


class CategoricalSampler(Sampler):
    def forward(self, logits):
        return torch.multinomial(logits.exp(), self.n_samples)
