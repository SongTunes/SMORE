import torch
import torch.nn as nn
import numpy as np

from mcs.nets.lower.decoder import DecoderCell

from mcs.nets.lower.encoder import AttentionEncoder, generate_state_hidden_transformer
from mcs.nets.functions import get_log_likelihood
from tsptw.functions import judge
from config import arg_parser


CFG = arg_parser()
TIME_SCALE = 0.01
TINY = 1e-6


class AttentionModel(nn.Module):

    def __init__(self, embed_dim=128, n_heads=8, tanh_clipping=10.):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping)
        self.max_n_couriers = CFG.max_n_couriers
        self.max_n_customers = CFG.max_n_customers
        self.episode_time = CFG.episode_time
        self.move_speed = CFG.move_meter_per_min
        self.sensing_time = CFG.task_time

        self.linear_state = nn.Linear(2 * embed_dim, embed_dim)
        self.linear_budget = nn.Linear(1, 32)
        self.linear_context = nn.Linear(embed_dim, embed_dim)

        self.encoder = AttentionEncoder()

    def forward(self, x, encoder_output, rest_budget, cs_state, ts_state, depot_grids, mask, state_context, couriers_selected=None, decode_type='greedy'):
        """

        @param x:
            x[0] -- depot_data: (batch, 1, 4) --> embed_depot_xy: (batch, embed_dim)
            x[1] -- grids_data: (batch, (grid_size)*n_timeslots), 4)
            x[2] -- customer_data: (batch, 1, 2 * max_n_customers)
            x[3] -- this_tsp_data: (batch)
            x[4] -- budget: (batch)
        @param encoder_output:
        @param rest_budget: [bs]
        @param cs_state:
        @param ts_state:
        @param mask: sensing task graph mask  [bs, n_sensingtasks+1, 1]
        @param couriers_selected:
        @param decode_type:

        @return:

        state.static: [bs, 1, 2*self.max_n_customers]
        state.dynamic: [bs, t, 1]  t<max_n_sensingtasks
        state.mask_fs: [bs, 1, max_n_customers]  float mask
        current_cost: [bs, max_n_couriers]
        """

        batch_size, n_sensingtasks, _ = x[1].size()

        all_mask = (rest_budget.unsqueeze(1).repeat(1, n_sensingtasks) + TINY < torch.gather(cs_state.task_cost, 1, couriers_selected[:, :, None].repeat(1, 1, n_sensingtasks)).squeeze(1)).unsqueeze(2)
        all_mask = torch.cat([torch.ones(batch_size, 1, 1, device=self.device).bool(), all_mask], dim=1)
        all_mask = all_mask + mask

        current_cost = cs_state.current_cost[:, None, :].repeat(1, n_sensingtasks, 1).reshape(batch_size*n_sensingtasks,
                                                                                              self.max_n_couriers)  # [b, max_n_couriers]
        # task_cost: [bs, max_nc, n_s]
        this_task_cost = torch.gather(cs_state.task_cost, 1, couriers_selected[:, :, None].repeat(1, 1, n_sensingtasks)).squeeze(1)  # [bs, n_s]
        this_current_cost = torch.gather(cs_state.current_cost, 1, couriers_selected).repeat(1, n_sensingtasks)  # [bs, 1]
        this_cost = (this_current_cost + this_task_cost).reshape(batch_size*n_sensingtasks) # [b]=bs*n_s
        new_cost = current_cost.scatter(1, couriers_selected[:, None, :].repeat(1, n_sensingtasks, 1).reshape(batch_size*n_sensingtasks, 1),
                                        this_cost.unsqueeze(1))  # current_cost: [b, max_n_couriers]

        #
        #
        #
        node_embeddings, graph_embedding, customer_embedding = encoder_output
        # state.static = customer_embedding

        total_tours = []
        total_logps = []

        # step_context = self.env.generate_state_hidden(ts_state, node_embeddings)
        step_context = generate_state_hidden_transformer(ts_state, depot_grids, node_embeddings, self.encoder)
        step_context = self.linear_state(step_context)  # [bs, 1, embed_dim]
        # budget scale
        h_budget = self.linear_budget((0.01 * rest_budget).unsqueeze(1).unsqueeze(2))
        state_context = self.linear_context(state_context)
        step_context = torch.cat([step_context, state_context, h_budget], dim=2)  # [bs, 1, embed_dim+1]

        task_cost = torch.gather(cs_state.task_cost, 1, couriers_selected[..., None].repeat(1, 1, n_sensingtasks))  # [bs, 1, n_s]
        task_value = cs_state.task_value
        dynamic_input = (task_cost, task_value)
        decoder_output = \
            self.Decoder(all_mask, step_context, encoder_output, dynamic_input, decode_type=decode_type)
        #

        #
        tours, logps = decoder_output  # [bs, 1] [bs, 1, n_sensingtasks+1]

        total_tours.append(tours)
        total_logps.append(logps)

        total_tours = torch.cat(total_tours, dim=1)  # [bs, 1]
        total_logps = torch.cat(total_logps, dim=1)  # [bs, 1, 441]

        new_cost = new_cost.reshape(batch_size, n_sensingtasks, self.max_n_couriers)
        cs_state.new_current_cost = torch.gather(new_cost, 1, torch.clamp(total_tours-1, min=0)[..., None].repeat(1, 1, self.max_n_couriers)).squeeze(1)
        # b = cs_state.new_current_cost.cpu().numpy()
        ll = get_log_likelihood(total_logps, total_tours)  # [bs]

        return ll, total_tours


def get_courier_mask(depot,
                     grids,
                     cs_state,
                     ts_state,
                     tsptw_solver,
                     budget,
                     this_tsp_data,
                     couriers_selected,
                     mask,
                     grid,
                     episode_time,
                     max_n_customers,
                     max_n_couriers,
                     move_speed,
                     device,
                     return_depot=True
                     ):
    """
    Based on the selected worker and his/her state,
    for all sensing tasks,
    judge if it is feasible for him/her under the current rest budget.

    @param depot:
    @param grids:
    @param cs_state:
    @param ts_state:
    @param tsptw_solver:
    @param budget:
    @param this_tsp_data:
    @param couriers_selected:
    @param mask:
    @param grid:
    @param episode_time:
    @param max_n_customers:
    @param max_n_couriers:
    @param move_speed:
    @param device:
    @return:
        Return 1: all_mask of this courier. {Sensing Task Graph Mask} + {TSPTW Time Mask} + {Total Cost > Budget Mask}
        [bs, n_sensingtasks+1, 1]  bool tensor
        means if the sensing task is feasible for the selected worker

        Return 2: new_cost of this courier. The Cost Distribution if we select a sensing task.
        [b, max_n_couriers] b=bs*n_sensingtasks
        means the new budget cost of the worker if complete a sensing task

    the value of new_cost is not trust-worthy where all_mask is True

    """

    batch_size, n_sensingtasks, _ = grids.size()
    b = batch_size * n_sensingtasks
    t = ts_state.dynamic.size(1)

    #
    depot_grids = torch.cat([depot, grids], dim=1).clone()
    # depot_grids = torch.cat([torch.zeros(batch_size, 1, 4, device=device), grids], dim=1).clone()
    this_tsp_data = this_tsp_data[..., None].repeat(1, n_sensingtasks).reshape(b)
    current_cost = cs_state.current_cost[:, None, :].repeat(1, n_sensingtasks, 1).reshape(b, max_n_couriers)  # [b, max_n_couriers]
    couriers_selected = couriers_selected[:, None, :].repeat(1, n_sensingtasks, 1).reshape(b, 1)  # [b, 1]

    #
    customer_data = ts_state.static.reshape(batch_size, max_n_customers, 2).clone()
    zero_index, target, target_tw, target4 = None, None, None, None
    if return_depot is False:
        zero_index = torch.argmax((customer_data.sum(2) == 0).float(), dim=1)  # [bs]
        target = torch.gather(customer_data, 1, torch.clamp(zero_index-1, min=0)[:, None, None].repeat(1, 1, 2))  # [bs, 1, 2]
        target_tw = torch.zeros(batch_size, 1, 2, device=device)
        target_tw[:, :, 1] = 0.004 * episode_time
        target4 = torch.cat([target, target_tw], dim=2)  # [bs, 1, 4]

    # concat a [0, episode_time] time window for custormer_data
    a = torch.zeros(batch_size, max_n_customers, 2, device=device)
    a[:, :, 1] = 0.004 * episode_time * (~(ts_state.mask_fs.bool())).squeeze(1).float()
    a[:, 0, 1] = 0.004 * episode_time  # depot remains
    customer_data4 = torch.cat([customer_data, a], dim=2)  # [bs, max_n_customers, 4]

    sensing_data = torch.gather(
        depot_grids[:, :, None, :].expand(batch_size, n_sensingtasks + 1, 1, 4),
        dim=1,
        index=ts_state.dynamic[..., None].clone().contiguous().expand(batch_size, t, 1, 4)
    ).squeeze(2)
    # [bs, t, 4]
    # mask_f: [bs, t, 1]
    m = ts_state.mask_f.squeeze(2).bool()
    sensing_data[:, :, 3] *= (~m)  # sensing tasks those are masked

    #
    y = depot.squeeze(1).repeat(n_sensingtasks, 1).clone()  # [b, 4]
    ye = target4.squeeze(1).repeat(n_sensingtasks, 1).clone() if not return_depot else None  # [b, 4]
    y_all = torch.zeros(b, t + max_n_customers + 1, 4, device=device)
    tsptw_mask = torch.zeros(b, t + max_n_customers + 1, device=device)
    # tsptw_mask: mask the state pad and customers pad

    for bs in range(batch_size):
        tsptw_mask[bs * n_sensingtasks: (bs + 1) * n_sensingtasks, t:t + max_n_customers] = ts_state.mask_fs[bs][0]
        tsptw_mask[bs * n_sensingtasks: (bs + 1) * n_sensingtasks, 0:t] = ts_state.mask_f[bs].squeeze(1)
        if return_depot is False:
            tsptw_mask[bs * n_sensingtasks: (bs + 1) * n_sensingtasks,  t:t + max_n_customers].scatter_(1, torch.clamp(zero_index-1, min=0)[bs][None, None].repeat(n_sensingtasks, 1), -np.inf)  # mask the target

        y_all[bs * n_sensingtasks: (bs + 1) * n_sensingtasks, t:t + max_n_customers, :] = customer_data4[bs, :, :]
        if return_depot is False:
            y_all[bs * n_sensingtasks: (bs + 1) * n_sensingtasks, t:t + max_n_customers, :].scatter_(1, torch.clamp(zero_index-1, min=0)[bs][None, None, None].repeat(n_sensingtasks, 1, 4), 0)
        y_all[bs * n_sensingtasks: (bs + 1) * n_sensingtasks, 0:t, :] = sensing_data[bs, :, :]
        y_all[bs * n_sensingtasks: (bs + 1) * n_sensingtasks, -1, :] = grids[bs]

        # state.mask_f: [bs, t, 1]
    #
    #

    reward_per_min = CFG.reward_per_min
    # scale
    y[:, 2:4] = y[:, 2:4] / 0.004 * TIME_SCALE
    if not return_depot:
        ye[:, 2:4] = ye[:, 2:4] / 0.004 * TIME_SCALE
    y_all[:, :, 2:4] = y_all[:, :, 2:4] / 0.004 * TIME_SCALE

    #
    #
    #
    if return_depot:
        judge_res, this_time = judge(tsptw_solver, y, y_all, tsptw_mask, grid, move_speed, device, return_depot)  # [b], [b]
    else:
        judge_res, this_time = judge(tsptw_solver, y, y_all, tsptw_mask, grid, move_speed, device, return_depot, xe=ye)  # [b], [b]
    #
    # if out of memory, maybe for bs in batch_size
    #
    # judge_res_list, this_time_list = [], []
    # for bs in range(batch_size):
    #
    #     judge_res_bs, this_time_bs = judge(tsptw_solver, y[bs*n_sensingtasks:(bs+1)*n_sensingtasks],
    #                                        y_all[bs*n_sensingtasks:(bs+1)*n_sensingtasks],
    #                                        tsptw_mask[bs*n_sensingtasks:(bs+1)*n_sensingtasks],
    #                                        grid, move_speed, device)  # [n_s], [n_s]
    #     judge_res_list.append(judge_res_bs)
    #     this_time_list.append(this_time_bs)
    # judge_res = torch.cat(judge_res_list, dim=0)
    # this_time = torch.cat(this_time_list, dim=0)

    #
    #
    this_time *= 100  # scale
    # print(time.time()-st)
    judge_res = judge_res.bool()
    # here, masked tasks those cannot complete
    this_cost = (this_time - (this_tsp_data / move_speed)) * reward_per_min  # [b]
    new_cost = current_cost.scatter(1, couriers_selected,
                                    this_cost.unsqueeze(1))  # current_cost: [b, max_n_couriers]

    new_total_cost = new_cost.sum(1)  # [b]

    judge_res += ~(budget.repeat(n_sensingtasks)+TINY >= new_total_cost)
    # here, masked tasks those can complete but budget exceed

    #
    mask1 = torch.cat([torch.ones((batch_size, 1, 1), device=device, dtype=torch.bool),
                       judge_res.reshape(batch_size, n_sensingtasks, 1)], dim=1)
    all_mask = mask + mask1  # [bs, n_sensingtasks+1, 1]

    return all_mask, new_cost


if __name__ == '__main__':
    pass
