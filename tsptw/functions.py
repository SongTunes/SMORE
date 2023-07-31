import numpy as np
import torch

from config import arg_parser


CFG = arg_parser()
DELIVERY_TIME = CFG.delivery_time
SENSING_TIME = CFG.task_time
EPISODE_TIME = CFG.episode_time


def str2bool(v):
    return v.lower() in ('true', '1')


def judge(solver, x, X, mask, grid, move_speed, device, return_depot=False, xe=None):
    """
    x, X: row&col * 0.1 and time * 0.01
    """

    mask = mask.clone()

    model_low, model_high = solver

    h_hid = None
    c_hid = None
    h = None
    c = None
    B_val, origin_size, _ = X.size()

    baseline = 0
    time_wait = torch.zeros(B_val, device=device)
    time_penalty = torch.zeros(B_val, device=device)
    total_time_cost = torch.zeros(B_val, device=device)
    total_time_penalty = torch.zeros(B_val, device=device)

    mask[:, 0] = -np.inf  # [b, full size]

    size = int((torch.max((~mask.bool()).float().sum(1), dim=0)[0] + 1).item())

    Enter = X[:, :, 2]  # Entering time
    Leave = X[:, :, 3]  # Leaving time

    y_cur = None
    y_ini = x.clone()
    y_pre = y_ini
    y_end = torch.zeros_like(x)
    if return_depot:
        y_tar = y_ini.clone()
    else:
        y_tar = xe.clone()

    #
    in_cycle = (mask.bool().sum(1) - origin_size).bool()  # ==0就是轮空的 不轮空的<0
    b_val_list = [i for i in range(B_val)]

    for k in range(size-1):
        with torch.no_grad():
            # st = time.time()
            _, h_hid, c_hid, latent = model_low(x=x, X_all=X, h=h_hid, c=c_hid, mask=mask, xe=xe)
            output, h, c, _ = model_high(x=x, X_all=X, h=h, c=c, mask=mask, latent=latent, xe=xe)
            # print(time.time()-st)

        idx = torch.argmax(output, dim=1)  # greedy baseline  [B_val]

        y_cur = X[b_val_list, idx.data].clone()

        #
        idx1 = torch.clamp(grid[y_cur[:, 0]*10, y_cur[:, 1]*10].unsqueeze(1) - 1, min=0)  # [bs, 1]
        idx2 = torch.clamp(grid[y_pre[:, 0]*10, y_pre[:, 1]*10].unsqueeze(1) - 1, min=0)  # [bs, 1]
        idx12 = idx2 + idx1 * grid.num_cols * grid.num_rows

        baseline = torch.gather(
            grid.dis_mat_t[1:, 1:].reshape(grid.num_rows*grid.num_cols*grid.num_rows*grid.num_cols)[None, ...].repeat(B_val, 1),  # [bs, nr*nc]
            dim=1,
            index=idx12.long()
        ).squeeze(1) / move_speed * 0.01
        # [bs]

        y_pre = y_cur.clone()
        x = X[b_val_list, idx.data].clone()

        total_time_cost += baseline * in_cycle.float()

        # enter time
        enter = Enter[b_val_list, idx.data]
        leave = Leave[b_val_list, idx.data]

        # determine the total reward and current enter time
        time_wait = torch.lt(total_time_cost, enter).float() * (
                    enter - total_time_cost)  # wait until the Enter Window open

        total_time_cost += time_wait * in_cycle.float()

        # after entering the time window, finish the sensing task or the delivery task
        # is_delivery = (leave - enter) == (EPISODE_TIME * 0.01)
        # has calculation value error of leave and enter
        is_delivery = torch.round((leave - enter)*10)/10 >= (EPISODE_TIME * 0.01)
        task_time = is_delivery * DELIVERY_TIME + (~is_delivery) * SENSING_TIME
        total_time_cost += task_time * 0.01 * in_cycle.float()

        time_penalty = torch.lt(leave, total_time_cost).float() * 10  # if leave exceed the tw_right, penalty
        # total_time_cost += time_penalty

        total_time_penalty += time_penalty * in_cycle.float()

        mask[b_val_list, idx.data] += -np.inf

        last_in_cycle = in_cycle.clone()
        in_cycle *= (mask.bool().sum(1) - origin_size).bool()  # ==0: blank run else: <0
        new_end = last_in_cycle ^ in_cycle
        y_end = y_end + new_end.float()[..., None].repeat(1, 4) * y_cur

        #
        if in_cycle.float().sum().item() == 0:
            break

    #
    idx1 = torch.clamp(grid[y_end[:, 0]*10, y_end[:, 1]*10].unsqueeze(1) - 1, min=0)  # [bs, 1]
    idx2 = torch.clamp(grid[y_tar[:, 0]*10, y_tar[:, 1]*10].unsqueeze(1) - 1, min=0)  # [bs, 1]
    idx12 = idx2 + idx1 * grid.num_cols * grid.num_rows

    tc = torch.gather(
                grid.dis_mat_t[1:, 1:].reshape(grid.num_rows*grid.num_cols*grid.num_rows*grid.num_cols)[None, ...].repeat(B_val, 1),  # [bs, nr*nc]
                dim=1,
                index=idx12.long()
            ).squeeze(1) / move_speed * 0.01
    total_time_cost += tc
    # [bs]

    # total_time_cost += torch.norm(y_cur[:, :2] - y_ini[:, :2], dim=1)

    return total_time_penalty, total_time_cost
    # total_time_penalty: >0 if not feasible [B_val]
    # total_time_cost: [B_val]


def get_sequence(solver, x, X, mask, grid, move_speed, device, return_depot=False, xe=None):
    """
    x, X: row&col * 0.1 and time * 0.01
    """
    seq = []


    mask = mask.clone()
    model_low, model_high = solver

    h_hid = None
    c_hid = None
    h = None
    c = None
    B_val, origin_size, _ = X.size()

    baseline = 0
    time_wait = torch.zeros(B_val, device=device)
    time_penalty = torch.zeros(B_val, device=device)
    total_time_cost = torch.zeros(B_val, device=device)
    total_time_penalty = torch.zeros(B_val, device=device)

    mask[:, 0] = -np.inf

    size = int((torch.max((~mask.bool()).float().sum(1), dim=0)[0] + 1).item())

    Enter = X[:, :, 2]  # Entering time
    Leave = X[:, :, 3]  # Leaving time

    y_cur = None
    y_ini = x.clone()
    y_pre = y_ini
    y_end = torch.zeros_like(x)
    if return_depot:
        y_tar = y_ini.clone()
    else:
        y_tar = xe.clone()

    in_cycle = (mask.bool().sum(1) - origin_size).bool()  # ==0: blank run else: <0

    for k in range(size - 1):
        with torch.no_grad():
            _, h_hid, c_hid, latent = model_low(x=x, X_all=X, h=h_hid, c=c_hid, mask=mask, xe=xe)
            output, h, c, _ = model_high(x=x, X_all=X, h=h, c=c, mask=mask, latent=latent, xe=xe)

        idx = torch.argmax(output, dim=1)  # greedy baseline  [B_val]
        seq.append(idx.item())
        y_cur = X[[i for i in range(B_val)], idx.data].clone()

        #
        idx1 = torch.clamp(grid[y_cur[:, 0] * 10, y_cur[:, 1] * 10].unsqueeze(1) - 1, min=0)  # [bs, 1]
        idx2 = torch.clamp(grid[y_pre[:, 0] * 10, y_pre[:, 1] * 10].unsqueeze(1) - 1, min=0)  # [bs, 1]
        idx12 = idx2 + idx1 * grid.num_cols * grid.num_rows

        baseline = torch.gather(
            grid.dis_mat_t[1:, 1:].reshape(grid.num_rows * grid.num_cols * grid.num_rows * grid.num_cols)[
                None, ...].repeat(B_val, 1),  # [bs, nr*nc]
            dim=1,
            index=idx12.long()
        ).squeeze(1) / move_speed * 0.01
        # [bs]

        y_pre = y_cur.clone()
        x = X[[i for i in range(B_val)], idx.data].clone()

        total_time_cost += baseline * in_cycle.float()

        # enter time
        enter = Enter[[i for i in range(B_val)], idx.data]
        leave = Leave[[i for i in range(B_val)], idx.data]

        # determine the total reward and current enter time
        time_wait = torch.lt(total_time_cost, enter).float() * (
                enter - total_time_cost)  # wait until the Enter Window open

        total_time_cost += time_wait * in_cycle.float()

        # after entering the time window, finish the sensing task or the delivery task
        # has calculation error too
        is_delivery = torch.round((leave - enter)*10)/10 >= (EPISODE_TIME * 0.01)
        task_time = is_delivery * DELIVERY_TIME + (~is_delivery) * SENSING_TIME
        total_time_cost += task_time * 0.01 * in_cycle.float()

        time_penalty = torch.lt(leave, total_time_cost).float() * 10
        # total_time_cost += time_penalty
        total_time_penalty += time_penalty * in_cycle.float()

        mask[[i for i in range(B_val)], idx.data] += -np.inf

        last_in_cycle = in_cycle.clone()
        in_cycle *= (mask.bool().sum(1) - origin_size).bool()  # ==0: blank run else: <0
        new_end = last_in_cycle ^ in_cycle
        y_end = y_end + new_end.float()[..., None].repeat(1, 4) * y_cur

        #
        if in_cycle.float().sum().item() == 0:
            break

    idx1 = torch.clamp(grid[y_end[:, 0] * 10, y_end[:, 1] * 10].unsqueeze(1) - 1, min=0)  # [bs, 1]
    idx2 = torch.clamp(grid[y_tar[:, 0] * 10, y_tar[:, 1] * 10].unsqueeze(1) - 1, min=0)  # [bs, 1]
    idx12 = idx2 + idx1 * grid.num_cols * grid.num_rows

    tc = torch.gather(
        grid.dis_mat_t[1:, 1:].reshape(grid.num_rows * grid.num_cols * grid.num_rows * grid.num_cols)[None, ...].repeat(
            B_val, 1),  # [bs, nr*nc]
        dim=1,
        index=idx12.long()
    ).squeeze(1) / move_speed * 0.01
    total_time_cost += tc
    # [bs]

    # total_time_cost += torch.norm(y_cur[:, :2] - y_ini[:, :2], dim=1)

    return seq, total_time_penalty


if __name__ == '__main__':
    pass
