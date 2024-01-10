import scipy.io as io
from torch.utils.tensorboard import SummaryWriter
from model.ptn_loader import sPtn, sPtn_TCA
from model.surrogate_act import SurrogateHeaviside
from model.snn import STCADenseLayer, SNN, STCAOutputLayer, ReadoutLayer
from params import *
import matplotlib.pyplot as plt
import Levenshtein
from torch import cosine_similarity as cos_sim
from sklearn.metrics import *
from functools import reduce


test_model_path = "./checkpoints/STCA-Last-91.41.t7"
colorset = ['blue', 'chocolate', 'darkgray', 'gold', 'green', 'hotpink', 'orange', 'purple', 'red', 'black', 'violet', 'tomato']


def getSpkLoader(path, batch_size=1):
    data = io.loadmat(path)
    data_ = sPtn_TCA(data['ptnTest'], data['test_labels'], data['TmaxTest'], TCA_slice, dt=dt)
    batch_data = torch.utils.data.DataLoader(data_, batch_size=batch_size, shuffle=True, num_workers=0)
    return batch_data


def plot_figure(vmem):
    plt.figure(1)
    for i in range(num_class):
        plt.plot(vmem[:, i].detach().cpu().numpy(), color=colorset[i])
    plt.show()


def over_thresh(vmem, tmax, vlastmem, multlabel=False):
    time_spike = torch.zeros(int(tmax)*5)
    time_bucket = torch.zeros(int(tmax)*5, num_class)
    spike_num = torch.zeros(num_class)
    for ineuron in range(num_class):
        v = vmem[:, ineuron]
        for i in range(len(v)):
            if v[i] >= 0:
                time_bucket[int(spike_num[ineuron])][ineuron] = i
                spike_num[ineuron] += 1
    # print(time_bucket)
    for ineuron in range(num_class):
        last_spike_time = time_bucket[0][ineuron]
        if last_spike_time == 0:
            continue
        if spike_num[ineuron] == 1:
            # print("neuron:", ineuron + 1, "Time:", last_spike_time)
            time_spike[int(last_spike_time)] = ineuron + 1
            continue
        inspike = 1
        for i in range(1, int(spike_num[ineuron])):
            inspike = 1
            differ = cos_sim(vlastmem[time_bucket[i][ineuron].long()], vlastmem[last_spike_time.long()], dim=0)
            # if time_bucket[i][ineuron]-last_spike_time > C:
            if differ <= C_sim:
                # print("neuron:", ineuron + 1, "Time:", last_spike_time)
                time_spike[int(last_spike_time)] = ineuron + 1
                inspike = 0
            last_spike_time = time_bucket[i][ineuron]
        if inspike == 1:
            # print("neuron:", ineuron + 1, "Time:", last_spike_time)
            time_spike[int(last_spike_time)] = ineuron + 1

    spike_list = []
    vmem_list = torch.zeros(tmax)
    idx_list = 0
    for i in range(int(tmax)):
        if time_spike[i] != 0:
        # if time_spike[i] != 0 and (time_spike[i] - 1) not in spike_list:
            spike_list.append(time_spike[i] - 1)
            vmem_list[idx_list] = vmem[i, int(time_spike[i] - 1)]
            idx_list += 1

    # func = lambda x, y: x if y in x else x + [y]
    # spike_list = reduce(func, [[], ] + spike_list)

    if not multlabel or idx_list == 4:
        return spike_list
    else:
        if len(spike_list) > 4:
            _, index = torch.sort(vmem_list, descending=True)
            new_spike = []
            for i in range(4):
                new_spike.append(spike_list[index[i]])
            return new_spike
        else:
            vmean = torch.mean(vmem, dim=0)
            _, index = torch.sort(vmean, descending=True)
            i = 0
            for _ in range(4-len(spike_list)):
                while index[i] in spike_list:
                    i += 1
                spike_list.append(index[i].cpu())
            return spike_list


def eval_result(pred, target):
    list1 = []
    list2 = []
    for i in range(len(pred)):
        list1.append(str(int(pred[i].numpy())))
    for i in range(len(target)):
        list2.append(str(int(target[i].numpy())))
    return Levenshtein.ratio(list1, list2)


def trans_labels(batch_size, num_neuron, labels, mode='train'):
    labels_t = torch.zeros(batch_size, num_neuron)
    if mode == 'test':
        for i in range(batch_size):
            labels_t[i][labels[i]] += 1
    else:
        for i in range(batch_size):
            for j in range(4):
                labels_t[i][labels[j][i]] += 1
    return labels_t


if __name__ == '__main__':
    layers = []

    spike_fn = SurrogateHeaviside.apply
    layers.append(STCADenseLayer(structure[0], structure[1], spike_fn, w_init_mean, w_init_std, recurrent=False,
                                 lateral_connections=False, fc_drop=fc_drop, return_vmem=True))
    # layers.append(ReadoutLayer(structure[1], structure[2], w_init_mean, w_init_std, time_reduction="volt"))
    layers.append(STCAOutputLayer(structure[1], structure[2], spike_fn, w_init_mean, w_init_std, recurrent=False, lateral_connections=False))
    model = SNN(layers).to(device, dtype)
    checkpoint = torch.load(test_model_path, map_location='cpu')
    print(checkpoint['best_acc'])
    # print(checkpoint['best_net'])
    model.load_state_dict(checkpoint['best_net'])
    val_ldr = getSpkLoader(test_path, batch_size=1)
    model.eval()
    TCA_ACC = 0
    it = 0
    for idx, (ptns, labels) in enumerate(val_ldr):
        ptns = ptns.permute([0, 2, 1]).contiguous().to(device, dtype)
        output_mem, vlast, _ = model(ptns)
        acc = 0
        # plot_figure(output_mem[0])
        pred = torch.zeros(output_mem.shape[0], num_class)
        label_t = trans_labels(output_mem.shape[0], num_class, labels)

        for i in range(output_mem.shape[0]):
            spike_output = over_thresh(output_mem[i], output_mem.shape[1], vlast[i], multlabel=False)
            label = []
            for j in range(TCA_slice):
                label.append(labels[j][i])
            for spike_idx in spike_output:
                pred[i][spike_idx.__int__()] += 1
            acc += eval_result(spike_output, label)
        if idx == 0:
            labels_tot = label_t[0]
            pred_tot = pred[0]
        else:
            labels_tot = torch.cat((labels_tot, label_t[0]))
            pred_tot = torch.cat((pred_tot, pred[0]))
        # print(pred, label_t)
        TCA_ACC += acc
        acc /= output_mem.shape[0]
        it += output_mem.shape[0]
        print("Batch:", idx, "TCA-Acc:", acc)
    f1 = round(f1_score(labels_tot, pred_tot, average='weighted'), 4)
    precision = round(precision_score(labels_tot, pred_tot, average='macro'), 4)
    recall = round(recall_score(labels_tot, pred_tot, average='macro'), 4)
    print("Final Acc:", round(TCA_ACC / it, 4))
    print('Precison: {0},\n F1: {1},\n Recall: {2}\n'.format(precision, f1, recall))

