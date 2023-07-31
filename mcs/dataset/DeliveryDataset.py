import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.Grid import Grid, MIN_LAT, MIN_LNG


DEPOT_INDEX = 43  # 200m


class CouriersDataSet(Dataset):
    def __init__(self, config, batch_size=None, device=None, path=None):
        super(CouriersDataSet, self).__init__()

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
        self.customers_data, self.tsps_data = self.get_customers_data(path)  # [161, 30, 50]
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
        self.depot_data = torch.zeros(self.num_samples, 4, 1)
        self.depot_data[:, 0, :] = self.grid.row_index(DEPOT_INDEX)
        self.depot_data[:, 1, :] = self.grid.col_index(DEPOT_INDEX)
        self.depot_data[:, 2, :] = 0.
        self.depot_data[:, 3, :] = self.episode_time

        #
        self.ntask_data = torch.zeros(self.num_samples, self.max_n_couriers, self.max_n_customers)
        self.ntask_data = self.ntask_data.to(self.device)

        #
        # transpose
        self.depot_data = self.depot_data.transpose(1, 2)
        self.grids_data = self.grids_data.transpose(1, 2)
        self.customers_data = self.customers_data.transpose(1, 2)  # [120, max_n_couriers, 2*15]
        self.tsps_data = self.tsps_data.transpose(1, 2)  # [161, max_n_couriers, 1]
        self.ntask_data = self.ntask_data.transpose(1, 2)
        self.customers_graph = torch.zeros(self.num_samples, self.max_n_couriers, 1, self.grid.num_rows, self.grid.num_cols)
        for i in range(self.num_samples):
            for j in range(self.max_n_couriers):
                for k in range(self.max_n_customers):
                    r = int(self.customers_data[i, j, 2*k].item())
                    c = int(self.customers_data[i, j, 2*k+1].item())
                    if r > 0 and c > 0:
                        self.customers_graph[i, j, 0, r-1, c-1] = 1
        #
        # scale
        self.depot_data[:, :, 0:2] = self.depot_data[:, :, 0:2] * 0.1  # 0~1.0 0~1.2
        self.depot_data[:, :, 2:4] = self.depot_data[:, :, 2:4] * 0.004  # 0~0.96
        self.grids_data[:, :, 0:2] = self.grids_data[:, :, 0:2] * 0.1  # 0~1.0 0~1.2
        self.grids_data[:, :, 2:4] = self.grids_data[:, :, 2:4] * 0.004  # 0~0.96
        self.customers_data[:, :, :] = self.customers_data[:, :, :] * 0.1  # 0~1.0 0~1.2

        #
        print('CouriersDataSet data preparation finished.')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.depot_data[idx], self.grids_data[idx], self.customers_data[idx], self.customers_graph[idx], self.tsps_data[idx], self.ntask_data[idx]

    #
    def get_index_data(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), self.config.data_path)
        else:
            path = os.path.join(os.path.dirname(__file__), path)

        record_pd = pd.read_csv(path, sep=';', usecols=[1, 2, 3, 4])
        record_pd['trip_id'] = pd.to_datetime(record_pd['trip_id'])

        record_pd = record_pd.set_index('trip_id')  # the 'trip_idx' dim disappear
        record_np = np.array(record_pd)  # num_records * max_deliveries
        # print(record_np)

        indexes_list = []  # element: record list which is composed of 2d tuples.
        tasknums_list = []
        max_num = 0
        max_len = 0
        ignore_num = 0

        for record in record_np:

            t2_list = []
            idx_list = []
            tasknum_list = []

            str2_list = record[2].split('|')
            for str2 in str2_list:
                l2 = str2.split(',')
                l2_digit = [float(e) for e in l2]
                t2 = tuple(l2_digit)
                t2_list.append(t2)
                idx_list.append(t2[0])
                tasknum_list.append(t2[1])

            if len(t2_list) > max_num:
                max_num = len(t2_list)
            # pad 0
            l = len(idx_list)
            if l > self.config.max_destination_num:
                ignore_num += 1
                continue
            # max_len = l if l > max_len else max_len
            while l < self.config.max_destination_num:
                idx_list.append(0.)
                tasknum_list.append(0.)
                l += 1
            indexes_list.append(idx_list)
            tasknums_list.append(tasknum_list)

        print('records num: ', len(indexes_list))
        print('ignore num: {}'.format(ignore_num))
        print('max nodes num: ', max_num)

        return indexes_list, tasknums_list

    def get_customers_data(self, path=None):
        """
        return shape: [self.num_samples, 2 * self.max_n_customers, self.max_n_couriers]
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), r'../../data/your-data-path')
        else:
            path = os.path.join(os.path.dirname(__file__), path)
        record_pd = pd.read_csv(path, sep=';', usecols=[1, 2, 3, 4, 5, 6, 7, 8])
        # sta_id;courier_id;trip_id;records;date;time;group_id
        record_np = np.array(record_pd)  # num_records * 7

        customer_list = []
        tsp_list = []

        current_customer_data = torch.zeros(1, 2 * self.max_n_customers, self.max_n_couriers)
        current_tsp_data = torch.zeros(1, 1, self.max_n_couriers)
        #
        n_groups = 1
        current_group_id = record_np[0][6]
        # print('group id: ', current_group_id)
        current_index = 0
        for record in record_np:

            this_group_id = record[6]

            if this_group_id != current_group_id:

                current_group_id = this_group_id
                # print('group id: ', current_group_id)
                current_index = n_groups * self.max_n_couriers
                n_groups += 1

                customer_list.append(current_customer_data)
                tsp_list.append(current_tsp_data)

                current_customer_data = torch.zeros(1, 2 * self.max_n_customers, self.max_n_couriers)
                current_tsp_data = torch.zeros(1, 1, self.max_n_couriers)

            courier_id = current_index % self.max_n_couriers
            str2_list = record[3].split('|')
            idx_list = []
            for str2 in str2_list:
                l2 = str2.split(',')
                l2_digit = [float(e) for e in l2]
                t2 = tuple(l2_digit)
                idx_list.append(t2[0])
            l = len(idx_list)
            rc_list = []
            for idx in idx_list:
                r, c = self.grid.row_index(idx), self.grid.col_index(idx)
                rc_list.append(r)
                rc_list.append(c)
            current_customer_data[0, 0:2*l, courier_id] = torch.Tensor(rc_list)
            current_tsp_data[0, 0, courier_id] = float(record[7])

            #
            current_index += 1
        #
        customer_list.append(current_customer_data)
        tsp_list.append(current_tsp_data)
        #
        customers_data = torch.cat(customer_list, dim=0)
        tsps_data = torch.cat(tsp_list, dim=0)
        print('n groups: ', n_groups)
        return customers_data, tsps_data  # [161, 2*max_n_customers, max_n_couriers]


if __name__ == '__main__':
    from config import load_pkl, train_parser
    cfg = load_pkl(train_parser().path)
    couriers_dataset = CouriersDataSet(cfg)
