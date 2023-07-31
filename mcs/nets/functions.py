import torch


def get_log_likelihood(_log_p, pi):
    """ _log_p: (batch, decode_step, n_nodes)
        pi: (batch, decode_step), predicted tour
    """
    log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
    return torch.sum(log_p.squeeze(-1), 1)
