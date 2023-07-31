import argparse
import numpy as np
import random
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from scipy.spatial import distance

from gpn import GPN
from config import arg_parser
from utils.Grid import Grid, MIN_LAT, MIN_LNG, MIN_LAT_MELB, MIN_LNG_MELB, START_LOCATION


CFG = arg_parser()
N_ROWS = CFG.n_rows
N_COLS = CFG.n_cols
GRID_LEN = 200.
MOVE_SPEED = CFG.move_meter_per_min
EPISODE_TIME = CFG.episode_time
TASK_DURATION = CFG.task_duration
TIME_SCALE = 0.01
MAX_N_CUSTOMERS = 10
DEPOT_ROW = 4
DEPOT_COL = 7
DELIVERY_TIME = CFG.delivery_time
SENSING_TIME = CFG.task_time
TINY0 = 0.1
return_depot = CFG.return_depot
grid = Grid(MIN_LAT, MIN_LNG, N_ROWS, N_COLS, 0.2, 0.2, use_real_roadnetwork=False)


def generate_data(B=512, size=50, return_depot=False):
    graph = np.random.rand(size, B, 2)
    # map to the grid
    graph[:, :, 0] *= N_ROWS
    graph[:, :, 1] *= N_COLS

    graph = np.ceil(graph)
    # depot
    if return_depot:
        graph[0, :, 0] = DEPOT_ROW
        graph[0, :, 1] = DEPOT_COL

    X = np.zeros([size, B, 4])  # xi, yi, ei, li, ci
    solutions = np.zeros(B)
    if return_depot:
        route = [x for x in range(size)] + [0]
    else:
        route = [x for x in range(size)]
    total_tour_len = 0

    for b in range(B):
        if return_depot:
            k_list = [e for e in range(1, size-1)]
        else:
            k_list = [e for e in range(1, size - 2)]
        n_delivery = random.randint(1, MAX_N_CUSTOMERS)
        # n_delivery = MAX_N_CUSTOMERS
        # n_delivery = 0
        delivery_list = random.sample(k_list, n_delivery)
        best = route.copy()
        # begin 2-opt
        graph_ = graph[:, b, :].copy()

        dmatrix = distance.cdist(graph_, graph_, 'euclidean')
        dmatrix = dmatrix * GRID_LEN / MOVE_SPEED * TIME_SCALE
        improved = True

        while improved:
            improved = False
            _size = size if return_depot else size-1
            for i in range(_size):
                for j in range(i + 2, _size + 1):
                    #   0 1 2 3 4 5 0
                    old_dist = dmatrix[best[i], best[i + 1]] + dmatrix[best[j], best[j - 1]]
                    new_dist = dmatrix[best[j], best[i + 1]] + dmatrix[best[i], best[j - 1]]
                    if new_dist < old_dist:
                        best[i + 1:j] = best[j - 1:i:-1]
                        # print(opt_tour)
                        improved = True


        cur_time = 0
        tour_len = 0
        X[0, b, :2] = graph_[best[0]]  # x0, y0
        X[0, b, 2] = 0  # e0 = 0
        X[0, b, 3] = EPISODE_TIME * TIME_SCALE
        #  X[0,b,4] = 0

        for k in range(1, size):
            # generate data with approximate solutions
            X[k, b, :2] = graph_[best[k]]  # xi, yi

            cur_time += dmatrix[best[k - 1], best[k]]
            tour_len += dmatrix[best[k - 1], best[k]]
            num1 = (cur_time / TIME_SCALE) // (TASK_DURATION)
            task_time = DELIVERY_TIME if k in delivery_list else SENSING_TIME
            if not return_depot and k == size - 1:
                task_time = 0.
            cur_time += task_time * TIME_SCALE
            tour_len += task_time * TIME_SCALE

            num2 = (cur_time / TIME_SCALE) // (TASK_DURATION)

            l = np.max([0, (num2) * (TASK_DURATION * TIME_SCALE)])
            h = (num2 + 1) * (TASK_DURATION * TIME_SCALE)  # allow to exceed
            # h = np.min([EPISODE_TIME * TIME_SCALE, (num2 + 1) * (TASK_DURATION * TIME_SCALE)])
            # assert h + TINY0 > l

            X[k, b, 2] = 0 if k in delivery_list or (not return_depot and k == size-1) else l
            X[k, b, 3] = EPISODE_TIME * TIME_SCALE if k in delivery_list or (not return_depot and k == size-1) else h
            if h > EPISODE_TIME * TIME_SCALE:
                X[k, b, 3] = h  # fix

            if k not in delivery_list and num1 < num2:  # waiting time add to cur_time
                cur_time = l + task_time * TIME_SCALE
                tour_len = l + task_time * TIME_SCALE

            assert cur_time <= h + TINY0, (l, cur_time, h)
            # assert cur_time - task_time * TIME_SCALE + TINY0 >= l, (l, cur_time - task_time * TIME_SCALE, h)

            # X[k,b,4] = cur_time    # indicate the optimal solution
        if return_depot:
            tour_len += dmatrix[best[size - 1], best[size]]
        solutions[b] += tour_len

    #
    if return_depot:
        X1 = X[0:1, :, :]
        X2 = X[1:, :, :]
        np.random.shuffle(X2)
        # [size-1, b, 4]

        # resort the delivery tasks
        for b in range(B):
            d_idxs = []
            for i in range(size-1):
                if X2[i, b, 2] == 0 and X2[i, b, 3] >= EPISODE_TIME * TIME_SCALE:  # delivery task
                    d_idxs.append(i)
            nd_idxs = [s for s in range(size-1) if s not in d_idxs]
            selected = X2[d_idxs, b, :]
            unselected = X2[nd_idxs, b, :]
            X2[:, b, :] = np.concatenate((selected, unselected), axis=0)


        # row and col
        X1[:, :, 0:2] = X1[:, :, 0:2] * 0.1
        X2[:, :, 0:2] = X2[:, :, 0:2] * 0.1

        X = np.concatenate((X1, X2), axis=0)
        X = X.transpose(1, 0, 2)  # [b, size, 4]
    else:
        X1 = X[0:1, :, :]
        X2 = X[1:-1, :, :]
        X3 = X[-1:, :, :]
        np.random.shuffle(X2)
        # [size-1, b, 4]

        # resort the delivery tasks
        for b in range(B):
            d_idxs = []
            for i in range(size - 2):
                if X2[i, b, 2] == 0 and X2[i, b, 3] >= EPISODE_TIME * TIME_SCALE:  # delivery task
                    d_idxs.append(i)
            nd_idxs = [s for s in range(size - 2) if s not in d_idxs]
            selected = X2[d_idxs, b, :]
            unselected = X2[nd_idxs, b, :]
            X2[:, b, :] = np.concatenate((selected, unselected), axis=0)

        # row and col
        X1[:, :, 0:2] = X1[:, :, 0:2] * 0.1
        X2[:, :, 0:2] = X2[:, :, 0:2] * 0.1
        X3[:, :, 0:2] = X3[:, :, 0:2] * 0.1

        X = np.concatenate((X1, X2, X3), axis=0)
        X = X.transpose(1, 0, 2)  # [b, size, 4]
    return X, solutions

'''
main
'''

parser = argparse.ArgumentParser(description="GPN with RL")
parser.add_argument('--size', default=15, help="size of TSPTW")
parser.add_argument('--epoch', default=500, help="number of epochs")
parser.add_argument('--batch_size', default=512, help='')
parser.add_argument('--train_size', default=2500, help='')
parser.add_argument('--val_size', default=1000, help='')
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
args = vars(parser.parse_args())

size = int(args['size'])
learn_rate = args['lr']  # learning rate
B = int(args['batch_size'])  # batch_size
B_val = int(args['val_size'])  # validation size
steps = int(args['train_size'])  # training steps
n_epoch = int(args['epoch'])  # epochs
save_root = './model/save-root'

print('=========================')
print('prepare to train low model')
print('=========================')
print('Hyperparameters:')
print('size', size)
print('learning rate', learn_rate)
print('batch size', B)
print('validation size', B_val)
print('steps', steps)
print('epoch', n_epoch)
print('save root:', save_root)
print('=========================')

# state = torch.load(save_root)
model = GPN(n_feature=4, n_hidden=128).cuda()
# model.load_state_dict(state['model'])

optimizer = optim.Adam(model.parameters(), lr=learn_rate)

lr_decay_step = 2500
lr_decay_rate = 0.96
opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step * 1000,
                                                          lr_decay_step), gamma=lr_decay_rate)

beta = 0.8
val_mean = []
val_std = []
val_accuracy = []

for epoch in range(n_epoch):
    for i in tqdm(range(steps)):

        optimizer.zero_grad()

        X, solutions = generate_data(B=B, size=size)
        Enter = X[:, :, 2]  # Entering time
        Leave = X[:, :, 3]  # Leaving time

        X = torch.Tensor(X).cuda()
        Enter = torch.Tensor(Enter).cuda()
        Leave = torch.Tensor(Leave).cuda()
        mask = torch.zeros(B, size).cuda()
        mask[:, 0] = -np.inf
        if not return_depot:
            mask[:, -1] = -np.inf

        R = 0
        logprobs = 0
        reward = 0

        time_wait = torch.zeros(B).cuda()
        time_penalty = torch.zeros(B).cuda()
        total_time_penalty = torch.zeros(B).cuda()
        total_time_cost = torch.zeros(B).cuda()
        total_time_wait = torch.zeros(B).cuda()

        # X = X.view(B,size,3)
        # Time = Time.view(B,size)

        x = X[:, 0, :]
        xe = X[:, -1, :]
        h = None
        c = None

        y_ini = x.clone()
        y_pre = y_ini
        if return_depot:
            y_end = x.clone()
        else:
            y_end = xe.clone()

        _size = size - 1 if return_depot else size - 2
        for k in range(_size):
            output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask, xe=xe)
            sampler = torch.distributions.Categorical(output)
            idx = sampler.sample()  # now the idx has B elements

            y_cur = X[[i for i in range(B)], idx.data].clone()

            reward = torch.norm(y_cur[:, :2] - y_pre[:, :2], dim=1) * 10 * GRID_LEN / MOVE_SPEED * TIME_SCALE

            y_pre = y_cur.clone()
            x = X[[i for i in range(B)], idx.data].clone()

            R += reward
            total_time_cost += reward

            # enter time
            enter = Enter[[i for i in range(B)], idx.data]
            leave = Leave[[i for i in range(B)], idx.data]

            # determine the total reward and current enter time
            time_wait = torch.lt(total_time_cost, enter).float() * (enter - total_time_cost)
            total_time_wait += time_wait  # total time cost
            total_time_cost += time_wait
            #
            is_delivery = (leave - enter) >= (EPISODE_TIME * 0.01)
            task_time = is_delivery * DELIVERY_TIME + (~is_delivery) * SENSING_TIME
            total_time_cost += task_time * TIME_SCALE

            time_penalty = torch.lt(leave, total_time_cost).float() * 10
            # total_time_cost += time_penalty
            total_time_penalty += time_penalty

            TINY = 1e-15
            logprobs += torch.log(output[[i for i in range(B)], idx.data] + TINY)

            mask[[i for i in range(B)], idx.data] += -np.inf
        #
        #
        R += torch.norm(y_cur[:, :2] - y_end[:, :2], dim=1) * 10 * GRID_LEN / MOVE_SPEED * TIME_SCALE
        total_time_cost += torch.norm(y_cur[:, :2] - y_end[:, :2], dim=1) * 10 * GRID_LEN / MOVE_SPEED * TIME_SCALE

        if i == 0:
            C = total_time_penalty.mean()
        else:
            C = (total_time_penalty * beta) + ((1. - beta) * total_time_penalty.mean())

        loss = ((total_time_penalty - C) * logprobs).mean()

        loss.backward()

        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_grad_norm, norm_type=2)
        optimizer.step()
        opt_scheduler.step()

        if i % 50 == 0:
            print("epoch:{}, batch:{}/{},  total time:{}, reward:{}, time:{}"
                  .format(epoch, i, steps, total_time_penalty.mean().item(),
                          R.mean().item(), total_time_wait.mean().item()))

            print("optimal upper bound:{}"
                  .format(solutions.mean()))

            X, solutions = generate_data(B=B_val, size=size)
            Enter = X[:, :, 2]  # Entering time
            Leave = X[:, :, 3]  # Leaving time

            X = torch.Tensor(X).cuda()
            Enter = torch.Tensor(Enter).cuda()
            Leave = torch.Tensor(Leave).cuda()
            mask = torch.zeros(B_val, size).cuda()
            mask[:, 0] = -np.inf
            if not return_depot:
                mask[:, -1] = -np.inf

            baseline = 0
            time_wait = torch.zeros(B_val).cuda()
            time_penalty = torch.zeros(B_val).cuda()
            total_time_cost = torch.zeros(B_val).cuda()
            total_time_penalty = torch.zeros(B_val).cuda()

            x = X[:, 0, :]
            xe = X[:, -1, :]
            h = None
            c = None
            y_ini = x.clone()
            y_pre = y_ini
            if return_depot:
                y_end = x.clone()
            else:
                y_end = xe.clone()
            _size = size - 1 if return_depot else size - 2
            for k in range(_size):
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask, xe=xe)
                idx = torch.argmax(output, dim=1)  # greedy baseline

                y_cur = X[[i for i in range(B_val)], idx.data].clone()

                baseline = torch.norm(y_cur[:, :2] - y_pre[:, :2], dim=1) * 10 * GRID_LEN / MOVE_SPEED * TIME_SCALE

                y_pre = y_cur.clone()
                x = X[[i for i in range(B_val)], idx.data].clone()

                total_time_cost += baseline

                # enter time
                enter = Enter[[i for i in range(B_val)], idx.data]
                leave = Leave[[i for i in range(B_val)], idx.data]

                # determine the total reward and current enter time
                time_wait = torch.lt(total_time_cost, enter).float() * (enter - total_time_cost)
                total_time_cost += time_wait

                #
                is_delivery = (leave - enter) >= (EPISODE_TIME * 0.01)
                task_time = is_delivery * DELIVERY_TIME + (~is_delivery) * SENSING_TIME
                total_time_cost += task_time * TIME_SCALE

                time_penalty = torch.lt(leave, total_time_cost).float() * 10
                # total_time_cost += time_penalty
                total_time_penalty += time_penalty

                mask[[i for i in range(B_val)], idx.data] += -np.inf

            accuracy = 1 - torch.lt(torch.zeros_like(total_time_penalty),
                                    total_time_penalty).sum().float() / total_time_penalty.size(0)
            print('validation result:{}, accuracy:{}'
                  .format(total_time_penalty.mean().item(), accuracy))

            val_mean.append(total_time_cost.mean().item())
            val_std.append(total_time_cost.std().item())
            val_accuracy.append(accuracy)

    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, save_root + str(epoch) + '_low.pt')
