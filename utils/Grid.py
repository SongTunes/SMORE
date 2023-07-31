import torch
import numpy as np
import requests
import json
import time
import traceback
import math

from utils.TPTK.common.spatial_func import SPoint, distance
from utils.TPTK.common.spatial_func import LAT_PER_METER, LNG_PER_METER
from config import arg_parser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_LAT = 39.87206896551724
MIN_LNG = 116.30278723404255
MIN_LAT_MELB = -37.85
MIN_LNG_MELB = 144.90
START_LAT = 39.8783
START_LNG = 116.3187
START_LOCATION = (39.8783, 116.3187)


class Grid:
    """
    index order
    30 31 32 33 34...
    20 21 22 23 24...
    10 11 12 13 14...
    01 02 03 04...
    """

    def __init__(self, min_lat, min_lng, num_rows, num_cols, km_per_cell_lat, km_per_cell_lng, use_real_roadnetwork=False):

        cfg = arg_parser()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        km_lat = num_rows * km_per_cell_lat
        km_lng = num_cols * km_per_cell_lng

        self.max_lat = min_lat + LAT_PER_METER * km_lat * 1000.0
        self.max_lng = min_lng + LNG_PER_METER * km_lng * 1000.0

        self.min_lat = min_lat
        self.min_lng = min_lng

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.km_per_cell_lat = km_per_cell_lat
        self.km_per_cell_lng = km_per_cell_lng

        self.m_per_cell_lat = km_per_cell_lat * 1000.
        self.m_per_cell_lng = km_per_cell_lng * 1000.

        # pre-calculate the distance matrix
        num_rc = num_rows * num_cols
        self.dis_mat = np.zeros((num_rc + 1, num_rc + 1))
        if not use_real_roadnetwork:
            for i in range(num_rc):
                for j in range(i, num_rc):
                    idx1 = i + 1
                    idx2 = j + 1
                    self.dis_mat[idx1][idx2] = self.dis_mat[idx2][idx1] = self.get_distance(idx1, idx2)
            # print('pre calculate grid distances finished.')
            self.dis_mat_t = torch.from_numpy(self.dis_mat)
            self.dis_mat_t = self.dis_mat_t.to(device)
        else:
            self.read_real_roadnetwork_distance('./real_dis_mat_bicycling.pt')

        #
        self.episode_time = cfg.episode_time
        self.task_duration = cfg.task_duration
        self.n_timeslots = int(self.episode_time // self.task_duration)

        # definitions of task value
        self.a = 0.2
        self.n_k = 3
        self.k_l_row = [1, 2, 5]
        self.k_l_col = [1, 2, 4]
        self.k_l_time = [1, 2, 4]
        self.k_n_row = [self.num_rows // l for l in self.k_l_row]  # [10, 5, 2]
        self.k_n_col = [self.num_cols // l for l in self.k_l_col]  # [12, 6, 3]
        self.k_n_time = [self.n_timeslots // l for l in self.k_l_time]
        #

    def __getitem__(self, item):
        """
        Transpose (row idx, col idx) to grid idx.
        :param item: 2d tuple (row index, col index)
        :return: index in grid
        """
        ri, ci = item

        _ri = ri - 1
        _ci = ci - 1

        res = _ri * self.num_cols + _ci + 1
        if torch.is_tensor(res):
            return torch.clamp(res, min=0)
        else:
            return max(res, 0)

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def reshape(self, km_per_cell_lat, km_per_cell_lng, num_rows=None, num_cols=None):

        if num_rows is None:
            num_rows = math.ceil((self.km_per_cell_lat / km_per_cell_lat) * self.num_rows)
        if num_cols is None:
            num_cols = math.ceil((self.km_per_cell_lng / km_per_cell_lng) * self.num_cols)
        km_lat = num_rows * km_per_cell_lat
        km_lng = num_cols * km_per_cell_lng

        self.max_lat = self.min_lat + LAT_PER_METER * km_lat * 1000.0
        self.max_lng = self.min_lng + LNG_PER_METER * km_lng * 1000.0

        # print(self.max_lat)
        # print(self.max_lng)

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.km_per_cell_lat = km_per_cell_lat
        self.km_per_cell_lng = km_per_cell_lng

        self.m_per_cell_lat = km_per_cell_lat * 1000.
        self.m_per_cell_lng = km_per_cell_lng * 1000.

    def center(self, idx):
        lat = (self.min_lat + LAT_PER_METER * self.m_per_cell_lat / 2) + LAT_PER_METER * (self.row_index(
            idx)-1) * self.m_per_cell_lat
        lng = (self.min_lng + LNG_PER_METER * self.m_per_cell_lng / 2) + LNG_PER_METER * (self.col_index(
            idx)-1) * self.m_per_cell_lng
        return lat, lng

    def row_index(self, idx):
        if idx == 0:
            return 0.
        return (idx - 1.) // self.num_cols + 1

    def col_index(self, idx):
        if idx == 0:
            return 0.
        return (idx - 1.) % self.num_cols + 1

    def rc2idx(self, ri, ci):
        """
        Inplace version of `get_item()`
        """
        ri -= 1
        ci -= 1
        return 1 + ri * self.num_cols + ci

    def mid(self, idx1, idx2):
        row_st = min(self.row_index(idx1), self.row_index(idx2))
        col_st = min(self.col_index(idx1), self.col_index(idx2))

        row_add = abs(self.row_index(idx1) - self.row_index(idx2)) // 2
        col_add = abs(self.col_index(idx1) - self.col_index(idx2)) // 2

        mid_idx = 1 + (row_st + row_add) * self.num_cols + (col_st + col_add)
        return mid_idx

    def get_distance(self, idx1, idx2):
        lat1, lng1 = self.center(idx1)
        lat2, lng2 = self.center(idx2)

        return distance(SPoint(lat1, lng1), SPoint(lat2, lng2))

    def distance(self, idx1, idx2):
        return self.dis_mat[int(idx1), int(idx2)]

    def index(self, lat, lng, preprocess=False):
        """

        :param lat: lat
        :param lng: lng
        :param preprocess: If preprocess, return -1 when ValueError instead of exit(1).
        :return:
        """
        # lat_interval = (self.max_lat - self.min_lat) / float(self.num_rows)
        # lng_interval = (self.max_lng - self.min_lng) / float(self.num_cols)
        try:
            # row_idx = int((lat - self.min_lat) // lat_interval)
            # col_idx = int((lng - self.min_lng) // lng_interval)
            if lat < self.min_lat or lat >= self.max_lat:
                raise ValueError('[Grid.index()] Location(' + str(lat) + ', ' + str(
                    lng) + ') not in the current grid. Check the lat.')
            if lng < self.min_lng or lng >= self.max_lng:
                raise ValueError('[Grid.index()] Location(' + str(lat) + ', ' + str(
                    lng) + ') not in the current grid. Check the lng.')

            delta_lat = lat - self.min_lat
            delta_lng = lng - self.min_lng

            m_lat = delta_lat / LAT_PER_METER
            m_lng = delta_lng / LNG_PER_METER
            row_idx = m_lat // self.m_per_cell_lat + 1
            col_idx = m_lng // self.m_per_cell_lng + 1
            return self.rc2idx(row_idx, col_idx)
        except ValueError:
            if not preprocess:
                traceback.print_exc()
                exit(1)
            else:
                return -1

    def coordinate(self, idx):
        """
        Transpose idx to lat and lng.
        :param idx:
        :return:
        """
        ri = self.row_index(idx)
        ci = self.col_index(idx)

        m_lat = (ri-1) * self.m_per_cell_lat + self.m_per_cell_lat / 2
        m_lng = (ci-1) * self.m_per_cell_lng + self.m_per_cell_lng / 2

        lat = self.min_lat + LAT_PER_METER * m_lat
        lng = self.min_lng + LNG_PER_METER * m_lng

        return lat, lng

    def up(self, idx):
        if self.row_index(idx) < self.num_rows - 1:
            return idx + self.num_cols
        return -1

    def down(self, idx):
        if self.row_index(idx) > 0:
            return idx - self.num_cols
        return -1

    def left(self, idx):
        if self.col_index(idx) > 0:
            return idx - 1
        return -1

    def right(self, idx):
        if self.col_index(idx) < self.num_cols - 1:
            return idx + 1
        return -1

    def lu(self, idx):
        if self.col_index(idx) > 0 and self.row_index(idx) < self.num_rows - 1:
            return self.up(self.left(idx))
        return -1

    def ld(self, idx):
        if self.col_index(idx) > 0 and self.row_index(idx) > 0:
            return self.down(self.left(idx))
        return -1

    def ru(self, idx):
        if self.col_index(idx) < self.num_cols - 1 and self.row_index(idx) < self.num_rows - 1:
            return self.up(self.right(idx))
        return -1

    def rd(self, idx):
        if self.col_index(idx) < self.num_cols - 1 and self.row_index(idx) > 0:
            return self.down(self.right(idx))
        return -1

    def get_kth_graph_value(self, graph, quantity):
        """
        Calculate the single kth graph value.
        @param graph: Tensor. shape: [bs, nr, nc, nt]
        @return: Tensor. shape: [bs]
        """
        bs, nr, nc, nt = graph.size()
        p = graph / (quantity[:, None, None, None].repeat(1, nr, nc, nt))
        log2p = torch.log2(p)
        entropy = -torch.where(log2p == -float("inf"), torch.zeros_like(p), p * log2p).sum(1).sum(1).sum(1)
        return entropy

    def get_graph_value(self, mask):
        """
        Return the current value of the hole grid graph.
        @param mask: Tensor. shape: [bs, n_sensingtasks]
        @return:
        """

        mask = mask.clone()
        bs, n_sensingtasks = mask.size()
        quantity = mask.float().sum(1)  # [bs]

        if quantity.sum().item() == 0.:
            return torch.zeros(bs, device=self.device)

        # reshape the mask
        graph = mask.reshape(bs, self.num_rows, self.num_cols, self.n_timeslots).float()
        #
        graph_1 = graph[...]

        graphs = [graph_1]
        # graph_3: [bs, 2, 3, 6]

        for k in range(1, self.n_k):
            graph_k = torch.zeros((bs, self.k_n_row[k], self.k_n_col[k], self.k_n_time[k]), device=self.device)
            for a in range(self.k_l_row[k]):
                for b in range(self.k_l_col[k]):
                    for c in range(self.k_l_time[k]):
                        graph_k += graph[:, a::self.k_l_row[k], b::self.k_l_col[k], c::self.k_l_time[k]]
            graphs.append(graph_k)
        entropy = torch.zeros(bs, device=self.device)

        for k in range(self.n_k):
            entropy_k = self.get_kth_graph_value(graphs[k], quantity)
            w_k = math.log2(self.k_n_row[0] * self.k_n_col[0] * self.k_n_time[0]) / \
                  math.log2(self.k_n_row[k] * self.k_n_col[k] * self.k_n_time[k])
            entropy += w_k * entropy_k / self.n_k
        #
        graph_value = 1 * (self.a * entropy + (1 - self.a) * torch.log2(1+quantity))  # [bs]
        # print(entropy, torch.log2(quantity))
        return graph_value

    def get_task_value(self, mask):
        """
        Calculate the current value of each task in current grid graph state.
        @param: mask: [bs, n_sensingtasks]
        @return: [bs, n_sensingtasks]
        """
        mask = mask.clone()
        bs, n_sensingtasks = mask.size()
        # [bs, n_sensingtasks, n_sensingtasks] -> [bs * n_sensingtasks, n_sensingtasks]
        mask = mask[:, None, :].repeat(1, n_sensingtasks, 1)
        before_mask = mask.clone()
        before_mask = before_mask.reshape(bs*n_sensingtasks, n_sensingtasks)

        mask[:, range(n_sensingtasks), range(n_sensingtasks)] = True
        after_mask = mask.reshape(bs*n_sensingtasks, n_sensingtasks)

        before_value = self.get_graph_value(before_mask)
        after_value = self.get_graph_value(after_mask)
        task_value = after_value - before_value  # [bs*n_sensingtasks]
        task_value = task_value.reshape(bs, n_sensingtasks)
        return task_value

    def get_real_roadnetwork_distance(self, save_path='./real_dis_mat.pt'):
        real_dis_mat = torch.clone(self.dis_mat_t)  # [1+n_s, 1+n_s]
        n_grids, _ = real_dis_mat.size()

        key = 'YOUR-API-KEY'
        mode = 'bicycling'  # walking driving(default)

        for i in range(1, n_grids):
            for j in range(1, n_grids):
                request_params = {'key': key, 'mode': mode}
                lat_from, lng_from = self.center(i)
                request_params['from'] = str(lat_from) + ',' + str(lng_from)

                lat_to, lng_to = self.center(j)
                request_params['to'] = str(lat_to) + ',' + str(lng_to)

                # print(request_params)

                request = requests.get(url='https://apis.map.qq.com/ws/distance/v1/matrix', params=request_params)
                # tencent max requests: 5 times per sec
                time.sleep(0.2)
                # status = json.loads(request.text).get('status')
                # message = json.loads(request.text).get('message')
                rows = json.loads(request.text).get('result').get('rows')  # list which element is dict
                row = rows[0]
                elements = row['elements']  # list

                real_dis_mat[i][j] = elements[0]['distance']
                print('from {} to {}: {}'.format(i, j, elements[0]['distance']))
            print('{} / {}'.format(i, n_grids))
            torch.save(real_dis_mat, save_path)

    def read_real_roadnetwork_distance(self, load_path='./real_dis_mat_bicycling.pt'):
        self.dis_mat_t = torch.load(load_path).clone()
        print(self.dis_mat_t)


if __name__ == '__main__':
    grid = Grid(MIN_LAT, MIN_LNG, 10, 12, 0.2, 0.2)
    # grid.read_real_roadnetwork_distance('./real_dis_mat_bicycling.pt')


