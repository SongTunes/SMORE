import torch
import torch.nn as nn
import numpy as np

from mcs.nets.upper.am import AttentionModel as AMU
from mcs.nets.lower.am import AttentionModel as AML
from mcs.nets.lower.am import get_courier_mask
from mcs.nets.encoder import GraphAttentionEncoder
from config import arg_parser


MAX_NUM = 99999


class CourierSelectingState:
    def __init__(self, batch_size, max_n_couriers, grid, static=None, static_h=None):
        cfg = arg_parser()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.batch_size = batch_size
        self.embed_dim = 128

        self.grid = grid
        self.return_depot = cfg.return_depot
        self.episode_time = cfg.episode_time
        self.move_speed = cfg.move_meter_per_min
        self.max_n_couriers = max_n_couriers
        self.max_n_customers = cfg.max_n_customers
        self.task_duration = cfg.task_duration
        self.n_timeslots = int(self.episode_time // self.task_duration)
        self.grid_size = cfg.n_rows * cfg.n_cols
        self.n_sensingtasks = int(self.grid_size * self.n_timeslots)

        #
        self.static_h = static_h if static_h is not None else torch.zeros([batch_size, max_n_couriers, self.embed_dim], device=self.device)
        self.static = static  # (batch, max_n_couriers, 2 * max_n_customers)
        self.dynamic = torch.zeros(batch_size, 1, self.max_n_couriers, device=self.device).long()  # id of assigned sensing tasks

        self.current_cost = torch.zeros(self.batch_size, self.max_n_couriers, device=self.device)
        self.new_current_cost = None

        self.has_task = torch.ones(self.batch_size, self.max_n_couriers, device=self.device).bool()
        #
        self.task_cost = torch.zeros(self.batch_size, self.max_n_couriers, self.n_sensingtasks, device=self.device)
        # task_cost: the cost (incentive difference) for a worker to complete a sensing task
        # set to inf if not able to complete
        # update after a sensing task-worker assignment

        self.task_cost[...] = MAX_NUM  # since inf * 0 = nan

        #
        # float tensor mask
        # can choose->0 else->-np.inf
        # the content is static
        # mask the 0 pad of travel task
        self.mask_fs = torch.zeros(batch_size, self.max_n_couriers, self.max_n_customers, device=self.device)
        assert static is not None
        for b in range(batch_size):
            for c in range(self.max_n_couriers):
                for i in range(self.max_n_customers):
                    self.mask_fs[b][c][i] = -np.inf if (static[b][c][2*i] == 0 and static[b][c][2*i+1] == 0) else 0
        self.mask_fs[:, :, 0] = -np.inf  # mask the initial flag
        #
        # mask the 0 pad of assigned sensing task
        self.mask_f = torch.zeros(batch_size, 1, self.max_n_couriers, device=self.device)
        # [bs, t, max_n_courier]
        self.mask_f[:, :] = -np.inf
        #
        self.task_value = torch.zeros(self.batch_size, self.n_sensingtasks, device=self.device)

    def update(self, couriers_selected, sensingtask_selected):
        """
        Update dynamic, mask_f
        @param couriers_selected: [bs, 1]
        @param sensingtask_selected: [bs, 1]
        @return:
        """

        add_state = self.dynamic[:, -1:, :].clone()  # [bs, 1, n_couriers]
        add_state[...] = 0.
        add_mask_f = self.mask_f[:, -1:, :].clone()  # [bs, 1, n_couriers]
        add_mask_f[...] = -np.inf

        couriers_selected = couriers_selected.unsqueeze(2)  # [bs, 1, 1]
        sensingtask_selected = sensingtask_selected.unsqueeze(2)

        add_state.scatter_(dim=2, index=couriers_selected, src=sensingtask_selected)
        add_mask_f.scatter_(2, couriers_selected, 0)

        self.dynamic = torch.cat([self.dynamic, add_state], dim=1)
        self.mask_f = torch.cat([self.mask_f, add_mask_f], dim=1)

        '''
        state: [bs, t, n_couriers]
        chosen_tour: [bs]
        add col: [bs, 1, n_couriers]
        new state: [bs, t+1, n_couriers]
        '''
        return

    def update_current_cost(self, in_cycle):
        self.current_cost = self.current_cost * ((~in_cycle)[..., None].repeat(1, self.max_n_couriers)) + \
                            self.new_current_cost * (in_cycle[..., None].repeat(1, self.max_n_couriers))

    def init_couriers_tasks_info(self, inputs, budget, mask, tsptw_solver, grid):
        """
        Init the task_cost and has_task for each Courier.
        @param inputs:
        @param budget:
        @param mask:
        @param tsptw_solver:
        @param grid:
        @return:
        """

        depot_data, grids_data, customers_data, customers_graph, tsps_data, _ = inputs

        for c in range(self.max_n_couriers):
            # Generate the couriers_selected
            couriers_selected = torch.tensor([c for _ in range(self.batch_size)]).unsqueeze(1)  # [bs, 1]
            couriers_selected = couriers_selected.to(self.device)

            # Generate the task selecting state according to the courier selected
            if not self.return_depot:
                this_depot_data = torch.gather(depot_data, 1, couriers_selected[:, :, None].repeat(1, 1, 4))
            else:
                this_depot_data = depot_data
            inputs_l = generate_aml_inputs(this_depot_data, grids_data, customers_data, couriers_selected, tsps_data, budget, self.max_n_customers)

            # ts_state: gather from cs_state based on the courier id
            ts_state = TaskSelectingState(self.batch_size, None, self.max_n_customers)
            ts_state.init(self, couriers_selected)

            self.update_couriers_tasks_info(inputs[2], inputs_l, ts_state, tsptw_solver, mask, grid, couriers_selected, budget)
        mask_workerpad = ~(inputs[2].clone().sum(2, keepdims=True).bool())  # [bs, max_n_couriers, 1]
        # self.task_cost += 1.
        self.task_cost = self.task_cost * (~mask_workerpad.repeat(1, 1, self.n_sensingtasks)) + mask_workerpad.repeat(1, 1, self.n_sensingtasks) * MAX_NUM  # pad couriers
        self.has_task = self.has_task * (~mask_workerpad).squeeze(2)
        # init the task value
        sensing_mask = mask[:, 1:, 0]  # [bs, n_sensingtasks] bool
        self.task_value = self.grid.get_task_value(sensing_mask)

    def update_couriers_tasks_info(self, customers_data, inputs_l, ts_state, tsptw_solver, mask, grid, couriers_selected, budget, sensingtask_selected=None):
        """

        @param customers_data:
        @param inputs_l:
        @param ts_state:
        @param tsptw_solver:
        @param sensingtask_mask:
        @param grid:
        @param couriers_selected:
        @param budget:
        @param sensingtask_selected:
        @return:
        """
        # update the state of the selected worker
        # print('** update ', ts_state.dynamic, ts_state.mask_f)
        all_mask, new_cost = get_courier_mask(
            depot=inputs_l[0],
            grids=inputs_l[1],
            cs_state=self,
            ts_state=ts_state,
            tsptw_solver=tsptw_solver,
            budget=inputs_l[4],
            this_tsp_data=inputs_l[3],
            couriers_selected=couriers_selected,
            mask=mask,
            grid=grid,
            episode_time=self.episode_time,
            max_n_customers=self.max_n_customers,
            max_n_couriers=self.max_n_couriers,
            move_speed=self.move_speed,
            device=self.device,
            return_depot=self.return_depot
        )  # [bs, n_sensingtasks+1, 1]  [b, max_n_couriers]

        #
        all_mask = all_mask.squeeze(2)[:, 1:]
        new_cost = new_cost.reshape(self.batch_size, self.n_sensingtasks, self.max_n_couriers).transpose(1, 2)

        # where all_mask is False: set the task_cost to new_cost
        v1 = all_mask * MAX_NUM + (new_cost.sum(1) - self.current_cost[..., None].repeat(1, 1, self.n_sensingtasks).sum(1)) * (~all_mask)  # [bs, n_sensingtasks]

        self.task_cost.scatter_(1, couriers_selected[..., None].repeat(1, 1, self.n_sensingtasks), v1.unsqueeze(1))  # 99999 or cost
        mask_workerpad = ~(customers_data.clone().sum(2, keepdims=True).bool())  # [bs, max_n_couriers, 1]
        self.task_cost = self.task_cost * (~mask_workerpad.repeat(1, 1, self.n_sensingtasks)) + \
                         mask_workerpad.repeat(1, 1, self.n_sensingtasks) * MAX_NUM

        rest_budget = budget - self.current_cost.sum(1)
        # print(rest_budget)
        v2 = rest_budget >= torch.min(
            torch.gather(self.task_cost, 1, couriers_selected[..., None].repeat(1, 1, self.n_sensingtasks)).squeeze(1),
            dim=1)[0]
        self.has_task.scatter_(1, couriers_selected, v2.unsqueeze(1))
        self.has_task = self.has_task * (~mask_workerpad).squeeze(2)

        if sensingtask_selected is not None:
            self.task_cost.scatter_(2, torch.clamp(sensingtask_selected-1, min=0)[:, None, :].repeat(1, self.max_n_couriers, 1), MAX_NUM)
            # -1: initial flag
            # update has_task
            for c in range(self.max_n_couriers):
                # print('budget: ', budget)
                # print('current_cost: ', self.current_cost)
                rest_budget = budget - self.current_cost.sum(1)
                couriers_selected = torch.tensor([c for _ in range(self.batch_size)], device=self.device).unsqueeze(1)  # [bs, 1]
                v2 = rest_budget >= torch.min(
                    torch.gather(self.task_cost, 1, couriers_selected[..., None].repeat(1, 1, self.n_sensingtasks)).squeeze(
                        1),
                    dim=1)[0]
                #
                self.has_task.scatter_(1, couriers_selected, v2.unsqueeze(1))
            mask_workerpad = ~(customers_data.clone().sum(2, keepdims=True).bool())  # [bs, max_n_couriers, 1]
            self.task_cost = self.task_cost * (~mask_workerpad.repeat(1, 1, self.n_sensingtasks)) + mask_workerpad.repeat(1, 1, self.n_sensingtasks) * 99999
            self.has_task = self.has_task * (~mask_workerpad).squeeze(2)
            # print('has_task: ', self.has_task)

            # update task value
            sensing_mask = mask[:, 1:, 0]
            self.task_value = self.grid.get_task_value(sensing_mask)


class TaskSelectingState:
    def __init__(self, batch_size, max_n_sensingtasks, max_n_customers, static=None, static_h=None, dynamic=None, mask_f=None, mask_fs=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.embed_dim = 128
        self.max_n_sensingtasks = max_n_sensingtasks
        self.max_n_customers = max_n_customers

        self.static = static
        self.static_h = static_h
        self.dynamic = dynamic
        self.mask_f = mask_f  # [bs, t, 1]
        self.mask_fs = mask_fs

    def update(self, tours, sensing_task_id):
        pass

    def init(self, cs_state, couriers_selected):

        embed_dim = self.embed_dim

        s = torch.gather(
            cs_state.static,  # (batch, max_n_couriers, 2*max_n_customers)
            dim=1,
            index=couriers_selected[..., None].repeat(1, 1, 2 * self.max_n_customers)
        )  # [bs, 1, 2*self.max_n_customers]
        sh = torch.gather(
            cs_state.static_h,  # (batch, max_n_couriers, embed_dim)
            dim=1,
            index=couriers_selected[..., None].repeat(1, 1, embed_dim)
        )  # [bs, 1, embed_dim]
        d = torch.gather(
            cs_state.dynamic,  # [bs, t, max_n_couriers]
            dim=2,
            index=couriers_selected[..., None].repeat(1, cs_state.dynamic.size(1), 1)
        )  # [bs, t, 1]
        mf = torch.gather(
            cs_state.mask_f,  # [bs, t, max_n_couriers]
            dim=2,
            index=couriers_selected[..., None].repeat(1, cs_state.dynamic.size(1), 1)
        )  # [bs, t, 1]
        mfs = torch.gather(
            cs_state.mask_fs,  # [bs, max_n_couriers, max_n_customers]
            dim=1,
            index=couriers_selected[..., None].repeat(1, 1, self.max_n_customers)
        )  # [bs, 1, max_n_customers]

        self.static = s
        self.static_h = sh
        self.dynamic = d
        self.mask_f = mf
        self.mask_fs = mfs

        return


class DRL(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.return_depot = cfg.return_depot

        self.batch_size = cfg.batch
        self.embed_dim = cfg.embed_dim

        self.total_budget = cfg.total_budget
        # self.rest_budget = cfg.total_budget
        self.episode_time = cfg.episode_time
        self.move_speed = cfg.move_meter_per_min  # m/min
        self.reward_per_min = cfg.reward_per_min

        self.max_n_couriers = cfg.max_n_couriers
        self.max_n_customers = cfg.max_n_customers
        self.n_sensingtasks = int((cfg.n_rows * cfg.n_cols) * (cfg.episode_time // cfg.task_duration))

        self.encoder = GraphAttentionEncoder()

        self.amu = AMU(cfg.embed_dim, cfg.n_heads, cfg.tanh_clipping)
        self.aml = AML(cfg.embed_dim, cfg.n_heads, cfg.tanh_clipping)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = None

    def generate_amu_inputs(self, depot_data, grids_data, customers_data, mask):
        """

        @param depot_data: [bs, 1, 4]
        @param grids_data: [bs, n_sensingtasks, 4]
        @param customers_data: [bs, max_n_couriers, 2*max_n_customers]
        @param mask: [bs, n_sensingtasks+1, 1]
        @return:
        """

        mask_no_depot = mask[:, 1:, :]  # [bs, n_sensingtasks, 1]
        _depot_data = torch.cat([depot_data, torch.ones(self.batch_size, 1, 1, device=self.device)], dim=2)
        _grids_data = torch.cat([grids_data, mask_no_depot.float()], dim=2)

        return _depot_data, _grids_data, customers_data

    def greedy_select(self, x, encoder_output, cs_state, mask_task, mask_worker):
        """

        @param x:
        @param encoder_output:
        @param cs_state:
            has_task: [bs, max_nc]
            task_cost: [bs, max_nc, n_s]
            task_value: [bs, n_s]
            mask_f: float(-inf, 0) [bs, 1, max_nc]
        @param mask_task: bool [bs, 1 + n_s, 1]
        @param mask_worker: bool [bs, max_nc, 1] mask the worker who has no task to choose
        @return:
            pair sequence: worker, task
        """
        node_embeddings, graph_embedding, customers_embedding = encoder_output
        # [bs, n_sensingtasks+1, embed_dim]  [bs, embed_dim]  [bs, max_n_couriers, embed_dim]
        cs_state.static_h = customers_embedding
        batch_size, n_sensingtasks_1, _ = node_embeddings.size()

        task_value = torch.cat([torch.zeros(batch_size, 1, device=self.device), cs_state.task_value], dim=1)
        task_value = task_value[:, None, :].repeat(1, self.max_n_couriers, 1)
        task_cost = torch.cat([torch.zeros(batch_size, self.max_n_couriers, 1, device=self.device), cs_state.task_cost], dim=2)
        # task_cost = torch.clamp(task_cost, min=0.1)

        task_value = torch.where(task_cost < 9999, task_value, torch.tensor(0., device=self.device))

        task_vc = (torch.clamp(task_value, min=0)) / (torch.clamp(task_cost, min=0) + 1e-6)
        # task_vc = torch.clamp(task_value, min=0)
        task_vc = task_vc.reshape(batch_size, self.max_n_couriers * n_sensingtasks_1)

        #
        mask_workerpad = ~(x[2].clone().sum(2, keepdims=True).bool())  # [bs, max_n_couriers, 1]
        mask_courier = mask_workerpad + mask_worker
        # pair mask
        mask = torch.zeros((batch_size, self.max_n_couriers, n_sensingtasks_1, 1), device=self.device).bool()
        mask[:, :, :, :] = mask_courier[:, :, None, :].repeat(1, 1, n_sensingtasks_1, 1)
        mask[:, :, :, :] = mask[:, :, :, :] + mask_task[:, None, :, :].repeat(1, self.max_n_couriers, 1, 1) + (task_cost > 9999)[..., None]
        mask = mask.reshape(batch_size, self.max_n_couriers * n_sensingtasks_1)
        mask_pair = mask[:, :]
        task_vc = -99999 * mask_pair + task_vc * (~mask_pair)

        v = task_value.cpu().numpy()
        c = task_cost.cpu().numpy()
        vc = task_value / task_cost

        max_info = torch.max(task_vc, dim=1)  # [bs, 1]
        pair_selected = max_info[1].unsqueeze(1)

        return pair_selected, v, c, vc

    def forward(self, inputs, grid, tsptw_solver, decode_type='greedy', epoch=-1):
        """

        @param inputs:
            x[0] -- depot_data: (batch, 1, 4) --> embed_depot_xy: (batch, embed_dim)
            x[1] -- grids_data: (batch, (grid_size)*n_timeslots), 4)
            x[2] -- customers_data: (batch, max_n_couriers, 2 * max_n_customers)
            x[3] -- tsps_data: (batch, max_n_couriers, 1)
            x[4] -- ntask_data:
        @param grid:
        @param tsptw_solver:
        @param decode_type:
        @param epoch:
        @return:
        """
        '''
        ****
        inputs of amu:
            courier set state:
                c0: assigned sensing task
                c1: assigned sensing task
                ...
                cc: assigned sensing task
            current sensing task graph:
                dim 4+1 to indicate task was selected or not
        outputs of amu:
            select a courier
            shape: [bs, 1]
        ****

        ****
        inputs of aml:
            sensing task graph: 
                dim 4
            amu selected courier id(outputs of amu):
                shape: [bs, 1]
            this courier's assigned sensing task

        outputs of aml:
            select a sensing task
        ****

        '''

        #
        depot_data, grids_data, customers_data, customers_graph, tsps_data, _ = inputs
        self.batch_size = depot_data.size(0)
        depot_grids = torch.cat([depot_data, grids_data], dim=1)

        mask = torch.zeros((self.batch_size, self.n_sensingtasks+1, 1), dtype=torch.bool).to(self.device)
        mask[:, 0, :] = True  # mask for sensing tasks

        budget = torch.tensor([self.total_budget for _ in range(self.batch_size)], device=self.device)

        # 1. Create cs_state and init task_cost and has_task
        cs_state = CourierSelectingState(self.batch_size, self.max_n_couriers, grid, static=customers_data)  # courier selecting state
        cs_state.init_couriers_tasks_info(
            inputs=inputs,
            budget=budget,
            mask=mask,
            tsptw_solver=tsptw_solver,
            grid=grid
        )
        #

        ll_u_list = []
        ll_l_list = []
        entropy_u_list = []

        mask_workerpad = ~(inputs[2].clone().sum(2, keepdims=True).bool())

        encoder_output = self.encoder(inputs, mask2=mask_workerpad)
        encoder_output_u = encoder_output
        encoder_output_l = encoder_output

        node_embeddings_u, graph_embedding_u, customers_embedding_u = encoder_output_u
        node_embeddings_u[:, 0, :] = torch.zeros((self.batch_size, self.embed_dim), device=self.device)
        encoder_output_u = (node_embeddings_u, graph_embedding_u, customers_embedding_u)

        node_embeddings_l, graph_embedding_l, customers_embedding_l = encoder_output_l
        node_embeddings_l[:, 0, :] = torch.zeros((self.batch_size, self.embed_dim), device=self.device)
        encoder_output_l = (node_embeddings_l, graph_embedding_l, customers_embedding_l)

        #
        # due to the worker data misalignment, we will blank run some instances utils every instance ends
        in_cycle = torch.ones(self.batch_size, device=self.device).bool()
        #
        rewards = torch.zeros(self.batch_size, device=self.device)
        #
        t = 1  # decoder time step
        while True:

            # 2. Select a worker
            rest_budget = budget - cs_state.current_cost.sum(1)

            ll_u, couriers_selected, state_context = self.amu(
                inputs,
                encoder_output_u,
                rest_budget,
                cs_state,
                mask_has_no_task=(~cs_state.has_task).unsqueeze(2),  # [bs, max_n_couriers, 1]
                decode_type=decode_type
            )  # ll_u: [bs]  couriers_selected: [bs, 1] stored the courier id
            # print('couriers_selected', couriers_selected)

            # generate the data for aml
            # Generate the task selecting state according to the courier selected
            if not self.return_depot:
                this_depot_data = torch.gather(depot_data, 1, couriers_selected[:, :, None].repeat(1, 1, 4))
            else:
                this_depot_data = depot_data
            inputs_l = generate_aml_inputs(this_depot_data, grids_data, customers_data, couriers_selected, tsps_data, budget, self.max_n_customers)

            # ts_state: gather from cs_state based on the selected courier id
            ts_state = TaskSelectingState(self.batch_size, None, self.max_n_customers)
            ts_state.init(cs_state, couriers_selected)

            # 3. Select a sensing task
            ll_l, sensingtask_selected = self.aml(
                inputs_l,
                encoder_output_l,
                rest_budget,
                cs_state,
                ts_state,
                depot_grids,
                mask,
                state_context=state_context,
                couriers_selected=couriers_selected,  # [bs, 1]
                decode_type=decode_type
            )
            # ll_l: [bs]  sensingtask_selected: [bs, 1]
            # all_mask: [bs, n_sensingtasks+1, 1] the selected task has not been update yet
            add_task_value = torch.gather(cs_state.task_value, 1, torch.clamp(sensingtask_selected-1, min=0)).squeeze(1)

            #
            # 4. Update process
            # update cs_state (dynamic, mask_f) based on the selected (worker, task) pair
            cs_state.update(couriers_selected, sensingtask_selected)
            # update cs_state.current_cost
            cs_state.update_current_cost(in_cycle)
            # update sensing task graph mask
            mask.scatter_(1, sensingtask_selected[..., None], True)

            # update the selected worker's task_cost and has_task
            # for unselected workers, set the task_cost to MAX_NUM at the corresponding index of selected task
            # and re-judge the has_task
            ts_state = TaskSelectingState(self.batch_size, None, self.max_n_customers)
            ts_state.init(cs_state, couriers_selected)

            #
            cs_state.update_couriers_tasks_info(
                customers_data=inputs[2],
                inputs_l=inputs_l,
                ts_state=ts_state,
                tsptw_solver=tsptw_solver,
                mask=mask,
                grid=grid,
                couriers_selected=couriers_selected,
                sensingtask_selected=sensingtask_selected,
                budget=budget
            )

            # 6. Collect info.s of this cycle
            # collect probabilities
            ll_u_list.append(ll_u.unsqueeze(1) * in_cycle.float().unsqueeze(1))
            ll_l_list.append(ll_l.unsqueeze(1) * in_cycle.float().unsqueeze(1))

            # collect rewards
            rewards += in_cycle.squeeze().float() * add_task_value
            # for instance not in cycle (blank run), the reward is 0

            # 7. Update info.s of next cycle
            in_cycle *= cs_state.has_task[:, 0:1].float().sum(1).bool()
            t += 1

            if in_cycle.float().sum().item() == 0:
                break

        #
        l1 = torch.cat(ll_u_list, dim=1)  # [bs, T]
        l2 = torch.cat(ll_l_list, dim=1)  # [bs, T]
        ll = torch.cat([l1, l2], dim=1).sum(1)  # [bs]

        return rewards, ll, cs_state


def generate_aml_inputs(depot_data, grids_data, customers_data, couriers_selected, tsps_data, budget, max_n_customers):
    """

    @param depot_data: [bs, 1, 4]
    @param grids_data: [bs, n_sensingtasks, 4]
    @param customers_data: [bs, max_n_couriers, 2*max_n_customers]
    @param couriers_selected: [bs, 1]
    @param tsps_data: [bs, max_n_couriers, 1]
    @param budget: [bs]
    @param max_n_customers:
    @return:
    """

    this_customers_data = torch.gather(
        customers_data,
        dim=1,
        index=couriers_selected[..., None].repeat(1, 1, 2*max_n_customers)
    )  # [bs, 1, 2*max_n_customers]

    this_tsp_data = torch.gather(
        tsps_data,  # (batch, max_n_couriers, 1)
        dim=1,
        index=couriers_selected[..., None]
    ).squeeze()  # [bs]

    return depot_data, grids_data, this_customers_data, this_tsp_data, budget


if __name__ == '__main__':
    pass
