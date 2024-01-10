import torch.nn as nn
import random
import torch
from torch import cosine_similarity as cos_sim


def trans_labels(batch_size, num_neuron, labels):
    labels_t = torch.zeros(batch_size, num_neuron)
    for i in range(batch_size):
        labels_t[i][labels[i]] = 1
    return labels_t


def sim_calc(volt_mat, ineuron):
    avg_loss = 0
    num_dec = 0
    volt_mat = volt_mat.T
    num_time_step = volt_mat.shape[0]
    base_vec = volt_mat[torch.argmax(volt_mat[:, ineuron])]
    for i in range(num_time_step):
        if volt_mat[i][ineuron] <= 0:
            num_dec = num_dec + 1
            continue
        curr_vec = volt_mat[i]
        sim = (cos_sim(base_vec, curr_vec, dim=0) + 1) / 2.0
        # sim = cos_sim(base_vec, curr_vec, dim=0)
        avg_loss = avg_loss + sim * volt_mat[i][ineuron]
    return avg_loss/(num_time_step - num_dec)


def sim_last_calc(volt_mat, volt_last_mat, ineuron):
    avg_loss = torch.tensor(0)
    num_dec = torch.tensor(0)
    volt_mat = volt_mat.T
    volt_last_mat = volt_last_mat.T
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


class STCA_Loss(nn.Module):
    def __init__(self, args):
        super(STCA_Loss, self).__init__()
        print('STCA_Loss')
        self.device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
        self.C = args.C
        self.C_sim = args.C_sim

    def forward(self, vmem, vlastmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        spike_output = torch.zeros(batch_size, num_neuron)
        labels_t = trans_labels(batch_size, num_neuron, labels)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= 0)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > self.C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0: num_cluster = 0
                end_list = torch.tensor(end_list, device=self.device)
                beg_list = torch.tensor(beg_list, device=self.device)
                # 没有脉冲簇而label为1
                spike_output[ibatch, ineuron] = num_cluster
                if labels_t[ibatch, ineuron] > (num_cluster > 0):
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - self.C, 0)
                        rmax = min(cluster_pos_spike[-1] + self.C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if torch.sum(mask == 0) <= 0:
                        loss = loss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]]
                    else:
                        loss = loss - torch.max(v[mask == 0])
                # 有脉冲簇而label为0
                if labels_t[ibatch, ineuron] < (num_cluster > 0):
                    mask = torch.zeros_like(v)
                    idx_cluster = torch.argmin(end_list - beg_list)
                    cluster_pos_spike = pos_spike[beg_list[idx_cluster]: end_list[idx_cluster] + 1]
                    mask[cluster_pos_spike[0]: cluster_pos_spike[-1] + 1] = 1
                    mask[v[0:num_time] <= 0] = 0
                    loss = loss + torch.mean(v[mask == 1])
        return loss, spike_output


class STCA_Last_Loss(nn.Module):
    def __init__(self, args):
        super(STCA_Last_Loss, self).__init__()
        print('STCA_Last_Loss')
        self.device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
        self.C = args.C
        self.C_sim = args.C_sim

    def forward(self, vmem, vlastmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        spike_output = torch.zeros(batch_size, num_neuron)
        labels_t = trans_labels(batch_size, num_neuron, labels)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= 0)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > self.C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0:
                    num_cluster = 0
                end_list = torch.tensor(end_list, device=self.device)
                beg_list = torch.tensor(beg_list, device=self.device)
                # 没有脉冲簇而label为1
                spike_output[ibatch, ineuron] = num_cluster
                if labels_t[ibatch, ineuron] > num_cluster:
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - self.C, 0)
                        rmax = min(cluster_pos_spike[-1] + self.C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if torch.sum(mask == 0) <= 0:
                        loss = loss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]]
                    else:
                        loss = loss - torch.max(v[mask == 0])
                # 有脉冲簇而label为0
                if labels_t[ibatch, ineuron] < num_cluster:
                    idx_cluster = torch.argmin(end_list - beg_list)
                    loss = loss + v[pos_spike[end_list[idx_cluster]]]
        return loss, spike_output


class Back_Layer_ATCA(nn.Module):
    def __init__(self, args):
        super(Back_Layer_ATCA, self).__init__()
        print('Back_Layer_ATCA')
        self.device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
        self.C = args.C
        self.C_sim = args.C_sim

    def forward(self, vmem, vlastmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        spike_output = torch.zeros(batch_size, num_neuron)
        labels_t = trans_labels(batch_size, num_neuron, labels)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= 0)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > self.C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0: num_cluster = 0
                end_list = torch.tensor(end_list, device=self.device)
                beg_list = torch.tensor(beg_list, device=self.device)
                # 没有脉冲簇而label为1
                spike_output[ibatch, ineuron] = num_cluster
                if labels_t[ibatch, ineuron] > num_cluster:
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - self.C, 0)
                        rmax = min(cluster_pos_spike[-1] + self.C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if torch.sum(mask == 0) <= 0:
                        loss = loss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]]
                    else:
                        loss = loss - torch.max(v[mask == 0])
                # 有脉冲簇而label为0
                if labels_t[ibatch, ineuron] < num_cluster:
                    # idx_cluster = torch.argmin(end_list - beg_list)
                    # volt_mat = vmem[ibatch, pos_spike[beg_list[idx_cluster]]: pos_spike[end_list[idx_cluster]] + 1, :]
                    # volt_last_mat = vlastmem[ibatch, pos_spike[beg_list[idx_cluster]]: pos_spike[end_list[idx_cluster]] + 1, :]
                    # loss = loss + sim_last_calc(volt_mat, volt_last_mat, ineuron)
                    avgloss = torch.tensor(0)
                    for icluster in range(num_cluster):
                        volt_mat = vmem[ibatch, :, pos_spike[beg_list[icluster]]: pos_spike[end_list[icluster]] + 1]
                        volt_last_mat = vlastmem[ibatch, :, pos_spike[beg_list[icluster]]: pos_spike[end_list[icluster]] + 1]
                        avgloss = avgloss + sim_last_calc(volt_mat, volt_last_mat, ineuron)
                    avgloss = avgloss / num_cluster
                    loss = loss + avgloss

        return loss, spike_output


class STCA_TCA_Loss(nn.Module):
    def __init__(self, args):
        super(STCA_TCA_Loss, self).__init__()
        print('STCA_TCA_Loss')
        self.device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
        self.C = args.C
        self.C_sim = args.C_sim

    def forward(self, vmem, vlastmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        labels_t = trans_labels(batch_size, num_neuron, labels)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        spike_output = torch.zeros(batch_size, num_neuron)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= 0)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > self.C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0:
                    num_cluster = 0
                end_list = torch.tensor(end_list, device=self.device)
                beg_list = torch.tensor(beg_list, device=self.device)
                # 没有脉冲簇而label为1
                spike_output[ibatch, ineuron] = num_cluster
                if labels_t[ibatch, ineuron] > num_cluster:
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - self.C, 0)
                        rmax = min(cluster_pos_spike[-1] + self.C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if torch.sum(mask == 0) <= 0:
                        loss = loss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]]
                    else:
                        loss = loss - torch.max(v[mask == 0])
                # 有脉冲簇而label为0
                if labels_t[ibatch, ineuron] < num_cluster:
                    mask = torch.zeros_like(v)
                    idx_cluster = torch.argmin(end_list - beg_list)
                    cluster_pos_spike = pos_spike[beg_list[idx_cluster]: end_list[idx_cluster] + 1]
                    mask[cluster_pos_spike[0]: cluster_pos_spike[-1] + 1] = 1
                    loss = loss + torch.max(v[mask == 1])
        return loss, spike_output


class ATCA_TCA_Loss(nn.Module):
    def __init__(self, args):
        super(ATCA_TCA_Loss, self).__init__()
        print('ATCA_TCA_Loss')
        self.device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
        self.C = args.C
        self.C_sim = args.C_sim

    def forward(self, vmem, vlastmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        spike_output = torch.zeros(batch_size, num_neuron)
        labels_t = trans_labels(batch_size, num_neuron, labels)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= 0)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                # 每一个簇在pos_spike中的开始和结束位置，闭区间
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > self.C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0:
                    num_cluster = 0
                end_list = torch.tensor(end_list, device=self.device)
                beg_list = torch.tensor(beg_list, device=self.device)
                # 没有脉冲簇而label为1
                spike_output[ibatch, ineuron] = num_cluster
                if labels_t[ibatch, ineuron] > num_cluster:
                    mask = torch.zeros_like(v)
                    for icluster in range(num_cluster):
                        cluster_pos_spike = pos_spike[beg_list[icluster]: end_list[icluster] + 1]
                        lmin = max(cluster_pos_spike[0] - self.C, 0)
                        rmax = min(cluster_pos_spike[-1] + self.C, num_time - 1)
                        mask[lmin: rmax + 1] = 1
                    if torch.sum(mask == 0) <= 0:
                        loss = loss + v[pos_spike[random.randint(0, pos_spike.numel() - 1)]]
                    else:
                        loss = loss - torch.max(v[mask == 0])
                # 有脉冲簇而label为0
                if labels_t[ibatch, ineuron] < num_cluster:
                    avgloss = torch.tensor(0)
                    for icluster in range(num_cluster):
                        volt_mat = vmem[ibatch, :, pos_spike[beg_list[icluster]]: pos_spike[end_list[icluster]] + 1]
                        volt_last_mat = vlastmem[ibatch, :, pos_spike[beg_list[icluster]]: pos_spike[end_list[icluster]] + 1]
                        avgloss = avgloss + sim_last_calc(volt_mat, volt_last_mat, ineuron)
                    avgloss = avgloss / num_cluster
                    loss = loss + avgloss
        return loss, spike_output


class Accumulate_Loss(nn.Module):
    def __init__(self, args):
        super(Accumulate_Loss, self).__init__()
        print('Accumulate_Loss')
        self.device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")

    def forward(self, vmem, vlastmem, labels):
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(vmem, labels).to(self.device)
        spike_out = []
        return loss, spike_out


class Mean_Loss(nn.Module):
    def __init__(self, args):
        super(Mean_Loss, self).__init__()
        print('Mean_Loss')
        self.device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")

    def forward(self, vmem, vlastmem, labels):
        loss_f = nn.CrossEntropyLoss()
        # print(vmem.shape)
        vmem = torch.mean(vmem, dim=2)
        # print(vmem.shape)
        loss = loss_f(vmem, labels).to(self.device)
        spike_out = []
        return loss, spike_out
