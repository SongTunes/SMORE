import torch
import torch.nn as nn

from mcs.nets.layers import MultiHeadAttention, DotProductAttention
from mcs.nets.Sampler import TopKSampler, CategoricalSampler


MAX_NUM = 99999
TINY = 1e-3


class DecoderCell(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, clip=10., **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.max_n_couriers = 45

        self.Wks1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wvs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wks2 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Wkd1 = nn.Linear(2, embed_dim, bias=False)
        self.Wvd = nn.Linear(2, embed_dim, bias=False)
        self.Wkd2 = nn.Linear(2, embed_dim, bias=False)

        self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_step = nn.Linear(embed_dim + embed_dim + 32, embed_dim, bias=False)

        self.MHA = MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, need_W=False)  # 多头注意力
        self.SHA = DotProductAttention(clip=clip, return_logits=True, head_depth=embed_dim)  # 单头注意力
        # SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads

    def compute_static(self, node_embeddings, graph_embedding):
        self.Q_fixed = self.Wq_fixed(graph_embedding[:, None, :])
        self.Ks1 = self.Wks1(node_embeddings)
        self.Vs = self.Wvs(node_embeddings)
        self.Ks2 = self.Wks2(node_embeddings)

    def compute_dynamic(self, mask, step_context, dynamic):
        Kd1 = self.Wkd1(dynamic)
        Vd = self.Wvd(dynamic)
        Kd2 = self.Wkd2(dynamic)

        K1 = self.Ks1 + Kd1
        K2 = self.Ks2 + Kd2
        V = self.Vs + Vd

        Q_step = self.Wq_step(step_context)
        Q1 = self.Q_fixed + Q_step  # q(c) [bs, 1, 128]
        Q2 = self.MHA([Q1, K1, V], mask=mask)
        Q2 = self.Wout(Q2)
        logits = self.SHA([Q2, K2, None], mask=mask)
        return logits.squeeze(dim=1)

    def forward(self, mask, step_context, encoder_output, dynamic, decode_type='sampling'):
        """

        @param dynamic:
        @param mask: bool [bs, 1+n_s, 1]
        @param step_context:
        @param encoder_output:
        @param decode_type:
        @return:
        """
        node_embeddings, graph_embedding, customers_embedding = encoder_output

        batch_size, n_sensingtasks_1, _ = node_embeddings.size()
        # node_embeddings: [bs, 1+n_s, 128]
        task_cost, task_value = dynamic
        task_cost = task_cost.clone().transpose(1, 2)  # [bs, n_s, 1]
        task_cost = torch.cat([torch.zeros((batch_size, 1, 1), device=self.device), task_cost], dim=1)  # [bs, 1+n_s, 1]
        task_cost[:, 0, 0] = MAX_NUM
        # task_cost = torch.where(task_cost < MAX_NUM, task_cost, torch.tensor(0., device=self.device))
        task_cost *= 0.01  # scale  [bs, 1+n_s, 1]

        task_value = task_value.clone().unsqueeze(2)  # [bs, n_s, 1]
        task_value = torch.cat([torch.zeros((batch_size, 1, 1), device=self.device), task_value],
                               dim=1)  # [bs, 1+n_s, 1]
        task_value = torch.where(task_cost < 0.01 * MAX_NUM, task_value, torch.tensor(0., device=self.device))

        dynamic_input = torch.cat([task_cost, task_value], dim=2)  # [bs, 1+n_s, 2]

        task_score = (torch.clamp(task_value, min=0)) / (torch.clamp(task_cost, min=0) + TINY)
        task_score = task_score * (~mask).float()
        # print(task_score)
        task_score = torch.clamp(task_score, max=5.)  # fix bug
        max_task_score = torch.max(task_score, dim=1, keepdim=True)[0]  # [bs, 1, 1]
        min_task_score = torch.min(task_score, dim=1, keepdim=True)[0]  # [bs, 1, 1]
        delta_score = (max_task_score - min_task_score)
        delta_score4div = torch.where(torch.abs(delta_score) <= 1e-5, torch.tensor(1., device=self.device), delta_score)
        task_score = (task_score - min_task_score) / delta_score4div  # [bs, ns, 1]
        task_score = torch.where(torch.abs(delta_score) <= 1e-5, torch.tensor(1., device=self.device), task_score)

        # node_embeddings_b = torch.cat([node_embeddings, task_cost.repeat(1, 1, 128)], dim=2)  # [bs, 1+n_s, 129]
        self.compute_static(node_embeddings, graph_embedding)

        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
        log_ps, tours = [], []

        logits = self.compute_dynamic(mask, step_context, dynamic_input)
        e1 = torch.exp(logits)
        e2 = torch.exp((-0.5) / (TINY + task_score.squeeze(2) ** 2))

        # p = e1
        p = e1 * e2
        p = p * (~mask).squeeze(2).float()
        p = torch.clamp(p, min=0)  # [bs, 961]

        sump = p.sum(1)  # [bs]
        sump = torch.where(sump <= 0, torch.tensor(1., device=self.device), sump)
        p = p / sump[:, None].repeat(1, n_sensingtasks_1)
        log_p = torch.log(torch.where(p <= 0, torch.tensor(1e-10, device=self.device), p))

        next_node = selecter(log_p)  # [bs, 1]

        # print(next_node)
        # print(torch.gather(p, 1, next_node))

        tours.append(next_node.squeeze(1))
        log_ps.append(log_p)

        tours = torch.stack(tours, 1)  # [bs, tour_len=1]
        log_ps = torch.stack(log_ps, 1)  # [bs, tour_len=1, n_nodes]

        return tours, log_ps


if __name__ == '__main__':
    pass
