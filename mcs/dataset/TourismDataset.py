import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.Grid import Grid, MIN_LAT, MIN_LNG


class TravelDataSet(Dataset):
    def __init__(self, config, batch_size=None, device=None, path=None):
        super(TravelDataSet, self).__init__()

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.episode_time = config.episode_time
        self.task_duration = config.task_duration
        assert self.episode_time % self.task_duration == 0
        self.n_timeslots = int(self.episode_time // self.task_duration)
        self.max_n_customers = config.max_n_customers
        self.max_n_couriers = config.max_n_couriers

        self.batch_size = config.batch if batch_size is None else batch_size
        seed = config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.grid = Grid(MIN_LAT, MIN_LNG, config.n_rows, config.n_cols,
                         config.km_per_cell_lat, config.km_per_cell_lng)
        # prepare the grids_data
        self.grid_size = self.config.n_rows * self.config.n_cols

        #
        # [bs, 2*max_n_customers, max_n_couriers]
        self.customers_data, self.tsps_data, self.depot_data = self.get_customers_data(path)  # [161, 30, 50]
        #
        self.num_samples = self.customers_data.shape[0]  # couriers group number
        print('num samples: ', self.num_samples)

        self.grids_data = torch.zeros(self.num_samples, 4, (self.grid_size) * self.n_timeslots)  # [bs, 4, grid_size*n_timeslots]
        # [0, :, grid_index 1 timeslot 1] [0, :, grid_index 1 timeslot 2], ...
        for i in range((self.grid_size) * self.n_timeslots):

            idx = i + 1
            grid_idx0 = i // self.n_timeslots
            grid_idx = grid_idx0 + 1

            loc = i
            self.grids_data[:, 0, loc] = self.grid.row_index(grid_idx)
            self.grids_data[:, 1, loc] = self.grid.col_index(grid_idx)
            self.grids_data[:, 2, loc] = (i % self.n_timeslots) * self.task_duration
            self.grids_data[:, 3, loc] = (i % self.n_timeslots) * self.task_duration + self.task_duration
        #
        # transpose
        self.depot_data = self.depot_data.transpose(1, 2)  # [n, max_nc, 4]
        self.grids_data = self.grids_data.transpose(1, 2)
        self.customers_data = self.customers_data.transpose(1, 2)  # [120, max_n_couriers, 2*15]
        # aa = self.customers_data.cpu().numpy()
        self.tsps_data = self.tsps_data.transpose(1, 2)  # [161, max_n_couriers, 1]

        self.customers_graph = torch.zeros(self.num_samples, self.max_n_couriers, 1, self.grid.num_rows, self.grid.num_cols)
        for i in range(self.num_samples):
            for j in range(self.max_n_couriers):
                for k in range(self.max_n_customers):
                    r = int(self.customers_data[i, j, 2*k].item())
                    c = int(self.customers_data[i, j, 2*k+1].item())
                    if k == 0:
                        self.customers_graph[i, j, 0, r-1, c-1] = 2
                    elif r > 0 and c > 0:
                        self.customers_graph[i, j, 0, r-1, c-1] = 1
                    else:
                        self.customers_graph[i, j, 0, r - 1, c - 1] = 3
                        break
        #
        # scale
        self.depot_data[:, :, 0:2] = self.depot_data[:, :, 0:2] * 0.1  # 0~1.0 0~1.2
        self.depot_data[:, :, 2:4] = self.depot_data[:, :, 2:4] * 0.004  # 0~0.96
        self.grids_data[:, :, 0:2] = self.grids_data[:, :, 0:2] * 0.1  # 0~1.0 0~1.2
        self.grids_data[:, :, 2:4] = self.grids_data[:, :, 2:4] * 0.004  # 0~0.96
        self.customers_data[:, :, :] = self.customers_data[:, :, :] * 0.1  # 0~1.0 0~1.2

        #
        print('TravelDataSet data preparation finished.')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.depot_data[idx], self.grids_data[idx], self.customers_data[idx], self.customers_graph[idx], self.tsps_data[idx], self.tsps_data[idx]

    def get_customers_data(self, path=None):
        """
        return shape: [self.num_samples, 2 * self.max_n_customers, self.max_n_couriers]
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), r'../../data/your-data-path')
        else:
            path = os.path.join(os.path.dirname(__file__), path)
        record_pd = pd.read_csv(path, sep=';', usecols=[1, 2, 3, 4])
        # sta_id;courier_id;trip_id;records;date;time;group_id
        record_np = np.array(record_pd)  # num_records * 7

        customer_list = []
        tsp_list = []
        depot_list = []

        current_customer_data = torch.zeros(1, 2 * self.max_n_customers, self.max_n_couriers)
        current_tsp_data = torch.zeros(1, 1, self.max_n_couriers)
        current_depot_data = torch.zeros(1, 4, self.max_n_couriers)
        #
        n_groups = 1
        current_group_id = record_np[0][1]
        # print('group id: ', current_group_id)
        current_index = 0
        for record in record_np:

            this_group_id = record[1]

            if this_group_id != current_group_id:

                current_group_id = this_group_id
                # print('group id: ', current_group_id)
                current_index = n_groups * self.max_n_couriers
                n_groups += 1

                customer_list.append(current_customer_data)
                tsp_list.append(current_tsp_data)
                depot_list.append(current_depot_data)

                current_customer_data = torch.zeros(1, 2 * self.max_n_customers, self.max_n_couriers)
                current_tsp_data = torch.zeros(1, 1, self.max_n_couriers)
                current_depot_data = torch.zeros(1, 4, self.max_n_couriers)

            courier_id = current_index % self.max_n_couriers
            str2_list = record[2].split('|')
            depot = float(str2_list[0])
            idx_list = []
            for k in range(0, len(str2_list)):
                idx_list.append(float(str2_list[k]))
            l = len(idx_list)
            rc_list = []
            for idx in idx_list:
                r, c = self.grid.row_index(idx), self.grid.col_index(idx)
                rc_list.append(r)
                rc_list.append(c)
            current_customer_data[0, 0:2*l, courier_id] = torch.Tensor(rc_list)
            current_tsp_data[0, 0, courier_id] = float(record[3])
            current_depot_data[0, 0, courier_id] = self.grid.row_index(depot)
            current_depot_data[0, 1, courier_id] = self.grid.col_index(depot)
            current_depot_data[0, 2, courier_id] = 0.
            current_depot_data[0, 3, courier_id] = self.episode_time

            #
            current_index += 1
        #
        customer_list.append(current_customer_data)
        tsp_list.append(current_tsp_data)
        depot_list.append(current_depot_data)
        #
        customers_data = torch.cat(customer_list, dim=0)
        tsps_data = torch.cat(tsp_list, dim=0)
        depot_data = torch.cat(depot_list, dim=0)
        print('n groups: ', n_groups)
        return customers_data, tsps_data, depot_data  # [161, 2*max_n_customers, max_n_couriers]


if __name__ == '__main__':
    from config import arg_parser
    cfg = arg_parser()
    couriers_dataset = TravelDataSet(cfg)
