from abc import abstractmethod, ABC
import torch
import numpy as np

from utils.TSP import TSP
from utils.Grid import Grid, MIN_LAT, MIN_LNG, MIN_LAT_MELB, MIN_LNG_MELB
from tsptw.functions import get_sequence
from mcs.nets.DRL import TaskSelectingState


TIME_SCALE = 0.01
DEPOT_INDEX = 43


class Solver(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.grid = Grid(MIN_LAT, MIN_LNG, config.n_rows, config.n_cols,
                         config.km_per_cell_lat, config.km_per_cell_lng)

        # self.grid = Grid(MIN_LAT_MELB, MIN_LNG_MELB, config.n_rows, config.n_cols,
        #                  config.km_per_cell_lat, config.km_per_cell_lng)

        self.n_rows = config.n_rows
        self.n_cols = config.n_cols
        self.grid_size = self.n_rows * self.n_cols
        self.km_per_cell_lat = config.km_per_cell_lat
        self.km_per_cell_lng = config.km_per_cell_lng

        self.total_budget = config.total_budget
        self.episode_time = config.episode_time
        self.reward_per_min = config.reward_per_min

        self.move_speed = config.move_meter_per_min
        self.delivery_time = config.delivery_time
        self.task_time = config.task_time

        self.task_duration = config.task_duration
        self.n_timeslots = int(self.episode_time // self.task_duration)

        self.max_n_couriers = config.max_n_couriers
        self.max_n_customers = config.max_n_customers
        self.n_sensingtasks = int(self.grid_size * self.n_timeslots)

        self.tsptw_solver = None

    def index_time2sensing_id(self, grid_index, visit_time):
        """
        (grid index, visit time) -> sensing task id
        Change the visit_time due to the waiting time maybe.
        @param grid_index:
        @param visit_time:
        @return:
        """
        id = (grid_index-1)*self.n_timeslots + visit_time // self.task_duration
        new_visit_time = visit_time
        if (visit_time - (visit_time // self.task_duration)) < self.task_time:  # wait
            new_visit_time = (visit_time // self.task_duration) + self.task_time

        return id, new_visit_time

    def rct2sensing_id(self, r, c, t):
        """

        @param r:
        @param c:
        @param t:
        @return: range: [0, n_sensingtasks)
        """
        return (self.grid[r, c] - 1) * self.n_timeslots + t

    def sensing_id2rct(self, sensing_id):
        if sensing_id >= self.n_sensingtasks:
            grid_index = sensing_id - self.n_sensingtasks
            r = self.grid.row_index(grid_index)
            c = self.grid.col_index(grid_index)
            t = -1
            return r, c, t
        t = sensing_id % self.n_timeslots
        grid_index = 1 + sensing_id // self.n_timeslots
        r = self.grid.row_index(grid_index)
        c = self.grid.col_index(grid_index)
        return r, c, t

    def sensing_id2grid_index(self, sensing_id):
        if sensing_id >= self.n_sensingtasks:
            return sensing_id - self.n_sensingtasks
        grid_index = 1 + sensing_id // self.n_timeslots
        return grid_index

    def get_route(self, n_couriers, cs_state, depot_data, grids_data, return_depot=True):
        def task_type(l, h):
            if l == 0. and h == 2.4:
                return 0  # delivery task
            return 1  # sensing task
        data_list = []
        batch_size = 1

        # vis
        for c in range(n_couriers):
            if not return_depot:
                this_depot_data = torch.gather(depot_data, 1, torch.tensor([c], device=self.device).long()[:, None, None].repeat(1, 1, 4))
            else:
                this_depot_data = depot_data
            depot_grids = torch.cat([this_depot_data[0:1], grids_data[0:1]], dim=1)
            print('Courier {}: '.format(c))
            print('* Route: ')
            print('* * Start Depot')
            ts_state = TaskSelectingState(1, None, self.max_n_customers)
            couriers_selected = torch.tensor([c], device=self.device).unsqueeze(1)
            ts_state.init(cs_state, couriers_selected)

            t = ts_state.dynamic.size(1)

            customer_data = ts_state.static.reshape(1, self.max_n_customers, 2)
            zero_index, target, target_tw, target4 = None, None, None, None
            if return_depot is False:
                zero_index = torch.argmax((customer_data.sum(2) == 0).float(), dim=1)  # [bs]
                target = torch.gather(customer_data, 1,
                                      torch.clamp(zero_index - 1, min=0)[:, None, None].repeat(1, 1, 2))  # [bs, 1, 2]
                target_tw = torch.zeros(batch_size, 1, 2, device=self.device)
                target_tw[:, :, 1] = 0.004 * self.episode_time
                target4 = torch.cat([target, target_tw], dim=2)  # [bs, 1, 4]

            a = torch.zeros(1, self.max_n_customers, 2,
                            device=self.device)  # 给customer_data拼接上[0, episode_time]时间窗
            a[:, :, 1] = 0.004 * self.episode_time * (~(ts_state.mask_fs.bool())).squeeze(1).float()  # mask掉的时间窗为[0, 0]
            a[:, 0, 1] = 0.004 * self.episode_time  # depot还是保持[0, 420]
            customer_data4 = torch.cat([customer_data, a], dim=2)  # [bs, max_n_customers, 4]

            sensing_data = torch.gather(
                depot_grids[:, :, None, :].expand(1, self.n_sensingtasks + 1, 1, 4),
                dim=1,
                index=ts_state.dynamic[..., None].clone().contiguous().expand(1, t, 1, 4)
            ).squeeze(2)

            m = ts_state.mask_f.squeeze(2).bool()
            sensing_data[:, :, 3] *= (~m)  # mask掉的sensing task时间窗为[0, 0]

            y = this_depot_data.squeeze(1).clone()  # [1, 4]
            if not return_depot:
                ye = target4.squeeze(1).clone()  # [b, 4]
            else:
                ye = None
            y_all = torch.zeros(1, t + self.max_n_customers, 4, device=self.device)
            tsptw_mask = torch.zeros(1, t + self.max_n_customers, device=self.device)

            tsptw_mask[:, t:t + self.max_n_customers] = ts_state.mask_fs[0][0]
            tsptw_mask[:, 0:t] = ts_state.mask_f[0].squeeze(1)
            if return_depot is False:
                tsptw_mask[:, t:t + self.max_n_customers].scatter_(1,torch.clamp(zero_index - 1,min=0)[0][None, None].repeat(1,1),-np.inf)  # mask the target

            y_all[:, t:t + self.max_n_customers, :] = customer_data4[0, :, :]
            y_all[:, 0:t, :] = sensing_data[0, :, :]
            if return_depot is False:
                y_all[:, t:t + self.max_n_customers, :].scatter_(1, torch.clamp(
                    zero_index - 1, min=0)[0][None, None, None].repeat(1, 1, 4), 0)

            y[:, 2:4] = y[:, 2:4] / 0.004 * TIME_SCALE
            if not return_depot:
                ye[:, 2:4] = ye[:, 2:4] / 0.004 * TIME_SCALE
            y_all[:, :, 2:4] = y_all[:, :, 2:4] / 0.004 * TIME_SCALE
            # [bs, t, c]

            if return_depot:
                seq, judge_res = get_sequence(self.tsptw_solver, y, y_all, tsptw_mask, self.grid, self.move_speed,
                                              self.device, return_depot)
            else:
                seq, judge_res = get_sequence(self.tsptw_solver, y, y_all, tsptw_mask, self.grid, self.move_speed, self.device, return_depot, xe=ye)
            # print(judge_res)
            y_all[:, :, 0:2] = y_all[:, :, 0:2] * 10  # recover row&col id
            y[:, 0:2] = y[:, 0:2] * 10  # recover row&col id
            if not return_depot:
                ye[:, 0:2] = ye[:, 0:2] * 10  # recover row&col id

            #
            #
            # print result
            DEPOT_LAT, DEPOT_LNG = self.grid.coordinate(DEPOT_INDEX)
            # row_list: [person, lat, lng, type]

            task_seq = []
            type_seq = []
            row_list = ["%02d" % c, DEPOT_LAT, DEPOT_LNG, 0]
            data_list.append(row_list)
            print('* * origin: {:d}'.format(int(self.grid[y[0][0].item(), y[0][1]].item())))
            task_seq.append(int(self.grid[y[0][0].item(), y[0][1]].item()))
            type_seq.append(0)
            for s in seq:
                grid_idx = int(self.grid[y_all[0][s][0].item(), y_all[0][s][1]].item())
                row, col, l, h = int(y_all[0][s][0]), int(y_all[0][s][1]), y_all[0][s][2], y_all[0][s][3]

                if grid_idx != DEPOT_INDEX:
                    lat, lng = self.grid.coordinate(grid_idx)
                    row_list = ["%02d" % c, lat, lng, task_type(l, h)]
                    data_list.append(row_list)
                    print('* * Grid[{:d}]({:d}, {:d}) Time Window({:.2f}, {:.2f})'.format(grid_idx, row, col, l, h))
                    task_seq.append(int(grid_idx))
            print('* * destination: {:d}'.format(int(self.grid[ye[0][0].item(), ye[0][1]].item())))
            if return_depot:
                task_seq.append(int(self.grid[y[0][0].item(), y[0][1]].item()))
            else:
                task_seq.append(int(self.grid[ye[0][0].item(), ye[0][1]].item()))
            type_seq.append(3)
            # print('* * End Depot')
            row_list = ["%02d" % c, DEPOT_LAT, DEPOT_LNG, 0]
            data_list.append(row_list)

    def calculate_tsp(self, move_sequences, return_depot=True):
        """

        @param return_depot:
        @param move_sequences: [n_courier, max_n_customers]
        @return:
            min_distances
            visit_seqs:
                List
                n_courier * n_customers(maybe different length of dim 2)
                grid index
        """
        min_distances = np.zeros((move_sequences.shape[0]))
        visit_seqs = []

        # calculate the min_distances mat
        for idx in range(move_sequences.shape[0]):
            sequence = move_sequences[idx]
            # print(sequence)
            num_nodes = np.where(sequence == 0)[0][0]
            if return_depot:
                dist_mat = np.zeros((num_nodes+1, num_nodes+1))
                for i in range(num_nodes+1):
                    for j in range(i, num_nodes+1):
                        dist_mat[i][j] = self.grid.distance(move_sequences[idx][i%num_nodes], move_sequences[idx][j%num_nodes])
                        dist_mat[j][i] = dist_mat[i][j]

                tsp = TSP(num_nodes+1, dist_mat)  # handle the tsp problem
                # print(num_nodes, sequence)
                min_distances[idx], visit_seq = tsp.solve()
                visit_seq[-1] = 0

                min_distances[idx] += (self.delivery_time * self.move_speed) * (len(visit_seq) - 2)
                visit_idx_seq = [int(sequence[e]) for e in visit_seq]
                visit_seqs.append(visit_idx_seq)
            else:
                dist_mat = np.zeros((num_nodes, num_nodes))
                for i in range(num_nodes):
                    for j in range(i, num_nodes):
                        dist_mat[i][j] = self.grid.distance(move_sequences[idx][i % num_nodes],
                                                            move_sequences[idx][j % num_nodes])
                        dist_mat[j][i] = dist_mat[i][j]

                tsp = TSP(num_nodes, dist_mat)  # handle the tsp problem
                # print(num_nodes, sequence)
                min_distances[idx], visit_seq = tsp.solve()
                # visit_seq[-1] = 0

                min_distances[idx] += (self.delivery_time * self.move_speed) * (len(visit_seq) - 2)
                visit_idx_seq = [int(sequence[e]) for e in visit_seq]
                visit_seqs.append(visit_idx_seq)

        return min_distances, visit_seqs

    @abstractmethod
    def solve(self):
        pass

