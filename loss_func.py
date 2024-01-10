import torch
import torch.nn as nn
import random
from params import *
from torch import cosine_similarity as cos_sim


def sim_last_calc(volt_mat, volt_last_mat, ineuron):
    avg_loss = 0
    num_dec = 0
    num_time_step = volt_mat.shape[0]
    base_vec = volt_last_mat[torch.argmax(volt_mat[:, ineuron])]
    for i in range(num_time_step):
        if volt_mat[i][ineuron] <= 0:
            num_dec = num_dec + 1
            continue
        curr_vec = volt_last_mat[i]
        # sim = (cos_sim(base_vec, curr_vec, dim=0) + 1) / 2.0
        sim = cos_sim(base_vec, curr_vec, dim=0)
        # print(sim)
        avg_loss = avg_loss + sim * volt_mat[i][ineuron]
    return avg_loss/(num_time_step - num_dec)


class STCA_TCA_Loss(nn.Module):
    def __init__(self):
        super(STCA_TCA_Loss, self).__init__()
        print('STCA_TCA_Loss')

    def forward(self, vmem, vlastmem, labels_t, ratio):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[2]
        num_time = vmem.shape[1]
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        spike_output = torch.zeros(batch_size, num_neuron)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, :, ineuron]
                pos_spike = torch.nonzero(v >= 0)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0:
                    num_cluster = 0
                end_list = torch.tensor(end_list, device=device)
                beg_list = torch.tensor(beg_list, device=device)
                # 没有脉冲簇而label为1
                spike_output[ibatch, ineuron] = num_cluster
                if labels_t[ibatch, ineuron] > num_cluster:
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - C, 0)
                        rmax = min(cluster_pos_spike[-1] + C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if torch.sum(mask == 0) <= 0:
                        loss = loss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]] + delta * ratio * num_cluster
                    else:
                        loss = loss - torch.max(v[mask == 0]) + delta * ratio * num_cluster
                # 有脉冲簇而label为0
                if labels_t[ibatch, ineuron] < num_cluster:
                    mask = torch.zeros_like(v)
                    idx_cluster = torch.argmin(end_list - beg_list)
                    cluster_pos_spike = pos_spike[beg_list[idx_cluster]: end_list[idx_cluster] + 1]
                    mask[cluster_pos_spike[0]: cluster_pos_spike[-1] + 1] = 1
                    loss = loss + torch.max(v[mask == 1]) + delta * ratio * num_cluster
        return loss, spike_output


class ATCA_TCA_Loss(nn.Module):
    def __init__(self):
        super(ATCA_TCA_Loss, self).__init__()
        print('ATCA_TCA_Loss')

    def forward(self, vmem, vlastmem, labels_t, ratio):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[2]
        num_time = vmem.shape[1]
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        spike_output = torch.zeros(batch_size, num_neuron)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, :, ineuron]
                pos_spike = torch.nonzero(v >= 0)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0:
                    num_cluster = 0
                end_list = torch.tensor(end_list, device=device)
                beg_list = torch.tensor(beg_list, device=device)
                # 没有脉冲簇而label为1
                spike_output[ibatch, ineuron] = num_cluster
                if labels_t[ibatch, ineuron] > num_cluster:
                    delta = labels_t[ibatch, ineuron] - num_cluster
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - C, 0)
                        rmax = min(cluster_pos_spike[-1] + C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if torch.sum(mask == 0) <= 0:
                        loss = loss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]]
                    else:
                        meanloss = torch.tensor(0)
                        for _ in range(delta.__int__()):
                            if torch.sum(mask == 0) <= 0:
                                meanloss = meanloss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]]
                                continue
                            time_max = torch.argmax(v)
                            min_time = max(time_max - C/2, 0).__int__()
                            max_time = min(time_max + C/2, num_time - 1).__int__()
                            meanloss = meanloss + torch.max(v[mask == 0])
                            mask[min_time: max_time + 1] = 1
                        meanloss = meanloss / delta
                        loss = loss - meanloss
                # 有脉冲簇而label为0
                if labels_t[ibatch, ineuron] < num_cluster:
                    avgloss = torch.tensor(0)
                    delta = num_cluster - labels_t[ibatch, ineuron]
                    record = torch.zeros(num_cluster)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        record[icluster] = torch.max(v[cluster_pos_spike[0]: cluster_pos_spike[-1] + 1])
                    record, _ = torch.sort(record)
                    for idx in range(delta.__int__()):
                        avgloss = avgloss + record[idx]
                    avgloss = avgloss / delta
                    loss = loss + avgloss
                    # for icluster in range(num_cluster):
                    #     volt_mat = vmem[ibatch, pos_spike[beg_list[icluster]]: pos_spike[end_list[icluster]] + 1, :]
                    #     volt_last_mat = vlastmem[ibatch,
                    #                     pos_spike[beg_list[icluster]]: pos_spike[end_list[icluster]] + 1, :]
                    #     avgloss = avgloss + sim_last_calc(volt_mat, volt_last_mat, ineuron)
                    # avgloss = avgloss / num_cluster
                    # loss = loss + avgloss
        return loss, spike_output