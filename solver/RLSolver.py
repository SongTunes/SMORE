import torch
from torch.utils.data import DataLoader

from solver.Solver import Solver
from mcs.nets.DRL import DRL
from mcs.dataset.DeliveryDataset import CouriersDataSet
from config import arg_parser
from tsptw.gpn import GPN


class RLSolver(Solver):
    def __init__(self, config, tsptw_solver, model_path=r'../mcs/model/your-model_path'):
        super(RLSolver, self).__init__(config)

        self.dataset = CouriersDataSet(config, batch_size=1, device=self.device,
                                       path=r'../data/your-data-path')
        # self.dataset = TravelDataSet(config, batch_size=1, device=self.device,
        #                                path=r'../data/your-data-path')
        self.return_depot = config.return_depot
        self.depot_row = None
        self.depot_col = None

        self.tsptw_solver = tsptw_solver

        self.model = DRL(config)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['actor'])
        self.model.to(self.device)
        self.model.eval()

    def solve(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        mean_reward = 0.
        for b, x in enumerate(dataloader):

            depot_data, grids_data, customers_data, customers_graph, tsps_data, ntasks_data = x
            depot_data = depot_data.to(device)
            grids_data = grids_data.to(device)
            customers_data = customers_data.to(device)
            customers_graph = customers_graph.to(device)
            tsps_data = tsps_data.to(device)
            ntasks_data = ntasks_data.to(device)
            x = (depot_data.to(device), grids_data.to(device), customers_data.to(device), customers_graph.to(device), tsps_data.to(device),
                      ntasks_data.to(device))

            #
            #
            reward, _, cs_state = self.model(x, self.grid, tsptw_solver=self.tsptw_solver, decode_type='greedy')
            mean_reward += reward
            # print('n tasks finished: ', reward)

            has_courier = customers_data[0].sum(1).bool()  # [max_n_couriers]
            n_couriers = has_courier.sum().long()

            self.get_route(n_couriers, cs_state, depot_data[0:1], grids_data, return_depot=self.return_depot)
        #
        mean_reward /= 40
        print('RL mean reward: ', mean_reward)
        return


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = arg_parser()

    tsptw_high = GPN(n_feature=4, n_hidden=128)
    tsptw_low = GPN(n_feature=4, n_hidden=128)

    #
    save_root_low = '../tsptw/model/your-tsptw-lower-model'
    save_root_high = '../tsptw/model/your-tsptw-upper-model'

    # save_root_low = '../tsptw/model/your-tsptw-lower-model'
    # save_root_high = '../tsptw/model/your-tsptw-upper-model'

    state_high = torch.load(save_root_high)
    tsptw_high.load_state_dict(state_high['model'])
    state_low = torch.load(save_root_low)
    tsptw_low.load_state_dict(state_low['model'])

    tsptw_high = tsptw_high.to(device)
    tsptw_low = tsptw_low.to(device)

    tsptw_high.eval()
    tsptw_low.eval()

    tsptw_solver = (tsptw_low, tsptw_high)

    rl_solver = RLSolver(cfg, tsptw_solver, model_path=r'../mcs/model/your-model_path')
    rl_solver.solve()
