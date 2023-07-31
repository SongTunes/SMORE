import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import matplotlib.pyplot as plt
import os

from mcs.nets.DRL import DRL
from config import arg_parser
from mcs.dataset.TourismDataset import TravelDataSet
from mcs.dataset.DeliveryDataset import CouriersDataSet
from tsptw.gpn import GPN
from mcs.nets.Critic import StateCritic


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(cfg):
    torch.backends.cudnn.benchmark = True
    log_path = cfg.log_dir
    if log_path is None:
        log_path = '%s%s_%s.csv' % (cfg.log_dir, cfg.task, cfg.dump_date)
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch, mode, time, batch, loss, est, reward\n')

    def rein_loss(model, critic, inputs, grid, tsptw_solver, decoder_type, train_type=True, epoch=-1):
        if train_type:
            rewards, ll, tours = model(inputs, grid, tsptw_solver, decode_type=decoder_type, epoch=epoch)
        else:
            with torch.no_grad():
                rewards, ll, tours = model(inputs, grid, tsptw_solver, decode_type=decoder_type, epoch=epoch)

        critic_est = critic(inputs).view(-1)
        advantage = rewards - critic_est

        actor_loss = (advantage.detach() * (-ll)).mean()
        critic_loss = torch.mean(advantage ** 2)

        # rewards ll: [bs]
        # b = bs[t] if bs is not None else baseline.eval(inputs, rewards, grid, tsptw_solver)
        return actor_loss, critic_loss, rewards.mean(), critic_est.mean()

    #
    tsptw_low = GPN(n_feature=4, n_hidden=128)
    tsptw_high = GPN(n_feature=4, n_hidden=128)

    if cfg.return_depot:
        save_root_low = r'your-tsptw-lower-model-path'
        save_root_high = r'your-tsptw-upper-model-path'
    else:
        save_root_low = r'your-tsptw-lower-model-path'
        save_root_high = r'your-tsptw-upper-model-path'

    state_high = torch.load(save_root_high)
    tsptw_high.load_state_dict(state_high['model'])
    state_low = torch.load(save_root_low)
    tsptw_low.load_state_dict(state_low['model'])

    tsptw_high.to(device)
    tsptw_low.to(device)

    tsptw_high.eval()
    tsptw_low.eval()

    tsptw_solver = (tsptw_low, tsptw_high)

    # checkpoint = torch.load(r'./model/your-checkpoint-path')

    #
    model = DRL(cfg)
    # model.load_state_dict(checkpoint['actor'])
    model.to(device)

    if cfg.return_depot:
        couriers_dataset = CouriersDataSet(cfg, path=r'../../data/your-train-path')
        couriers_dataset_valid = CouriersDataSet(cfg, path=r'../../data/your-valid-path')
        couriers_dataset_test = CouriersDataSet(cfg, path=r'../../data/your-test-path')
    else:
        couriers_dataset = TravelDataSet(cfg, path=r'../data/your-train-path')
        couriers_dataset_valid = TravelDataSet(cfg, path=r'../data/your-valid-path')
        couriers_dataset_test = TravelDataSet(cfg, path=r'../data/your-test-path')

    #
    critic = StateCritic()
    # critic.load_state_dict(checkpoint['critic'])
    critic.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # optimizer.load_state_dict(checkpoint['actor_optimizer'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.5 * cfg.lr)
    # critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.75, last_epoch=-1)
    # critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, 5, gamma=0.75, last_epoch=-1)

    t1 = time()
    epoch_list, train_reward_list, valid_reward_list, test_reward_list, est_list = [], [], [], [], []
    epoch = 0
    # epoch = (checkpoint['epoch']) + 1
    while epoch < cfg.epochs:

        epoch_list.append(epoch)
        model.train()
        critic.train()

        ave_loss, ave_L, ave_est = 0., 0., 0.

        #
        dataloader = DataLoader(couriers_dataset, batch_size=cfg.batch, shuffle=True)
        for t, inputs in enumerate(dataloader):
            depot_data, grids_data, customers_data, customers_graph, tsps_data, ntasks_data = inputs
            inputs = (
                depot_data.to(device), grids_data.to(device), customers_data.to(device), customers_graph.to(device),
                tsps_data.to(device), ntasks_data.to(device)
            )
            actor_loss, critic_loss, L_mean, est = rein_loss(
                model, critic, inputs, couriers_dataset.grid, tsptw_solver, 'sampling', train_type=True, epoch=epoch
            )
            loss = actor_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2.0, norm_type=2)
            critic_optimizer.step()

            ave_loss += actor_loss.item()
            ave_L += L_mean.item()
            ave_est += est.item()

            if t % (cfg.batch_verbose) == 0:
                t2 = time()
                print('Epoch %d (batch = %d): Est: %1.3f Loss: %1.3f L: %1.3f, %dmin%dsec' % (
                    epoch, t, ave_est / (t + 1), ave_loss / (t + 1), ave_L / (t + 1),
                    (t2 - t1) // 60, (t2 - t1) % 60))
                with open(log_path, 'a') as f:
                    f.write('%d, Train, %dmin%dsec, %d, %1.3f, %1.3f, %1.3f\n' % (
                        epoch, (t2 - t1) // 60, (t2 - t1) % 60, t, ave_loss / (t + 1), ave_est / (t + 1),
                        ave_L / (t + 1)))
                t1 = time()

        train_reward_list.append(ave_L / 6)
        est_list.append(ave_est / 6)
        #
        if epoch % (cfg.epoch_valid) == 0:
            model.eval()
            critic.eval()
            # valid
            ave_loss, ave_L, ave_est = 0., 0., 0.
            valid_dataloader = DataLoader(couriers_dataset_valid, batch_size=cfg.batch, shuffle=False)
            for vt, inputs in enumerate(valid_dataloader):
                depot_data, grids_data, customers_data, customers_graph, tsps_data, ntasks_data = inputs
                inputs = (
                    depot_data.to(device), grids_data.to(device), customers_data.to(device), customers_graph.to(device),
                    tsps_data.to(device),
                    ntasks_data.to(device)
                )

                _, _, L_mean, est = rein_loss(
                    model, critic, inputs, couriers_dataset_valid.grid, tsptw_solver, 'greedy', False
                )
                ave_L += L_mean.item()
                if vt % (cfg.batch_verbose) == 0:
                    t2 = time()
                    print('valid: Epoch %d (batch = %d): L: %1.3f, %dmin%dsec' % (
                        epoch, vt, ave_L / (vt + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                    with open(log_path, 'a') as f:
                        f.write('%d, Valid, %dmin%dsec, %d, %1.3f, %1.3f, %1.3f\n' % (
                            epoch, (t2 - t1) // 60, (t2 - t1) % 60, vt, ave_loss / (vt + 1), ave_est / (vt + 1),
                            ave_L / (vt + 1)))
                    t1 = time()

            valid_reward_list.append(ave_L / 2)

            #
            # test
            ave_loss, ave_L, ave_est = 0., 0., 0.
            test_dataloader = DataLoader(couriers_dataset_test, batch_size=cfg.batch, shuffle=False)
            for vt, inputs in enumerate(test_dataloader):
                depot_data, grids_data, customers_data, customers_graph, tsps_data, ntasks_data = inputs
                inputs = (
                    depot_data.to(device), grids_data.to(device), customers_data.to(device), customers_graph.to(device),
                    tsps_data.to(device),
                    ntasks_data.to(device)
                )

                _, _, L_mean, est = rein_loss(
                    model, critic, inputs, couriers_dataset_test.grid, tsptw_solver, 'greedy', False
                )
                ave_L += L_mean.item()
                if vt % (cfg.batch_verbose) == 0:
                    t2 = time()
                    print('Test: Epoch %d (batch = %d): L: %1.3f, %dmin%dsec' % (
                        epoch, vt, ave_L / (vt + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                    with open(log_path, 'a') as f:
                        f.write('%d, Test, %dmin%dsec, %d, %1.3f, %1.3f, %1.3f\n' % (
                            epoch, (t2 - t1) // 60, (t2 - t1) % 60, vt, ave_loss / (vt + 1), ave_est / (vt + 1),
                            ave_L / (vt + 1)))
                    t1 = time()

            test_reward_list.append(ave_L / 2)

        # scheduler.step()
        # critic_scheduler.step()

        draw_picture(epoch, epoch_list, train_reward_list, valid_reward_list, test_reward_list, est_list)
        checkpoint = {
            'epoch': epoch,
            'actor': model.state_dict(),
            'critic': critic.state_dict(),
            'actor_optimizer': optimizer.state_dict(),
            'critic_optimizer': critic_optimizer.state_dict()
        }
        torch.save(checkpoint, '%scheckpoint_epoch%s.pt' % (cfg.weight_dir, epoch))

        epoch += 1


def draw_picture(epoch, epoch_list, train_reward_list, valid_reward_list, test_reward_list, est_list):
    plt.clf()
    plt.figure(1)
    plt.plot(epoch_list, train_reward_list)
    plt.xlabel('epoch')
    plt.ylabel('train reward')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), './picture/' + "your-train-pic-path"))

    plt.clf()
    plt.figure(1)
    plt.plot(epoch_list, valid_reward_list)
    plt.xlabel('epoch')
    plt.ylabel('valid reward')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), './picture/' + "your-valid-pic-path"))

    plt.clf()
    plt.figure(1)
    plt.plot(epoch_list, test_reward_list)
    plt.xlabel('epoch')
    plt.ylabel('test reward')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), './picture/' + "your-test-pic-path"))

    plt.clf()
    plt.figure(1)
    plt.plot(epoch_list, est_list)
    plt.xlabel('epoch')
    plt.ylabel('est')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), './picture/' + "your-est-pic-path"))


if __name__ == '__main__':
    cfg = arg_parser()
    train(cfg)
