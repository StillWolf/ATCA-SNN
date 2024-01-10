from GetData import dvs_ges_tca
import argparse
import os
from torch.utils.data import DataLoader
from Resnet import *
import tqdm
from loss_func import *
import numpy as np
import Levenshtein
from matplotlib import pyplot as plt
from sklearn.metrics import *


colorset = ['blue', 'chocolate', 'darkgray', 'gold', 'green', 'hotpink', 'orange', 'purple', 'red', 'black', 'violet', 'tomato']
def over_thresh(vmem, tmax, thresh, interval):
    time_spike = torch.zeros(int(tmax)*5)
    time_bucket = torch.zeros(int(tmax)*5, 11)
    spike_num = torch.zeros(11)
    for ineuron in range(11):
        v = vmem[ineuron, :]
        for i in range(len(v)):
            if v[i] >= thresh:
                time_bucket[int(spike_num[ineuron])][ineuron] = i
                spike_num[ineuron] += 1
    # print(time_bucket)
    for ineuron in range(11):
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
            if time_bucket[i][ineuron]-last_spike_time > interval:
                # print("neuron:", ineuron + 1, "Time:", last_spike_time)
                time_spike[int(last_spike_time)] = ineuron + 1
                inspike = 0
            last_spike_time = time_bucket[i][ineuron]
        if inspike == 1:
            # print("neuron:", ineuron + 1, "Time:", last_spike_time)
            time_spike[int(last_spike_time)] = ineuron + 1

    spike_list = []
    for i in range(int(tmax)):
        if time_spike[i] != 0 and time_spike[i]-1 not in spike_list:
            spike_list.append(time_spike[i]-1)
    return spike_list


def eval_result(pred, target):
    list1 = []
    list2 = []
    for i in range(len(pred)):
        list1.append(str(int(pred[i].numpy())))
    for i in range(len(target)):
        list2.append(str(int(target[i].numpy())))
    # print(list1)
    # print(list2)
    return Levenshtein.seqratio(list1, list2)


def plot_vmem(vmem):
    plt.figure(0)
    for i in range(11):
        plt.plot(vmem[i].detach().cpu().numpy(), color=colorset[i])
    plt.savefig('./fig.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r'/data3/ql/DVSG/hdf5', type=str)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--C", default=8, type=int)
    parser.add_argument("--C_sim", default=0.97, type=int)
    parser.add_argument("--mode", default='ATCA', type=str)
    parser.add_argument("--seed", default=9853, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--vth", default=0.5, type=int)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

    device_ids = [0, 1, 2]
    device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
    test_data = dvs_ges_tca(args.root, device, 4, is_train=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    snn = Cifar_Net_NoBn(decay=False)
    snn = nn.DataParallel(snn, device_ids=device_ids)
    snn.to(device)
    model = torch.load('./checkpoints/CifarNet_ATCA_model_89.46.pth')
    print(model['max_test_acc'])
    new_state_dict = {k: v for k, v in model['net'].items() if k in snn.state_dict()}
    snn.load_state_dict(new_state_dict)
    snn.eval()

    acc_list = torch.zeros(10)
    f1_list = torch.zeros(10)
    precision_list = torch.zeros(10)
    recall_list = torch.zeros(10)

    for test_id in range(10):
        acc = 0
        pred_tot = []
        labels_tot = []
        length = test_loader.__len__()
        for i, (images, labels) in enumerate(tqdm.tqdm(test_loader)):
            back_noise = torch.tensor(np.random.normal(loc=0.0, scale=0.1, size=200))
            images += back_noise.to(device)
            spike_out, v_trace, v_trace_last = snn(images)
            pred_list = over_thresh(v_trace[0], 200, 0, args.C)
            acc += eval_result(pred_list, labels[0])
            label_spike = F.one_hot(labels[0], num_classes=11)
            label_spike = label_spike.sum(dim=0)
            pred_ls = torch.zeros(len(pred_list), dtype=torch.int64)
            for j in range(len(pred_list)):
                pred_ls[j] = pred_list[j]
            pred_ls = F.one_hot(pred_ls, num_classes=11)
            pred_ls = pred_ls.sum(dim=0)
            if i == 0:
                labels_tot = label_spike
                pred_tot = pred_ls
            else:
                labels_tot = torch.cat((labels_tot, label_spike))
                pred_tot = torch.cat((pred_tot, pred_ls))
            # plot_vmem(v_trace[0])
            # print(label_spike, pred_ls)
        f1 = round(f1_score(labels_tot, pred_tot, average='weighted'), 4)
        precision = round(precision_score(labels_tot, pred_tot, average='macro'), 4)
        recall = round(recall_score(labels_tot, pred_tot, average='macro'), 4)
        print('Precison: {0},\n F1: {1},\n Recall: {2}\n Acc: {3}'.format(precision, f1, recall, acc/length))
        f1_list[test_id] = f1
        precision_list[test_id] = precision
        recall_list[test_id] = recall
        acc_list[test_id] = acc/length
    print("Acc:", torch.mean(acc_list) * 100, "+-", torch.std(acc_list) * 100)
    print("Acc:", acc_list)
    print("F1:", torch.mean(f1_list) * 100, "+-", torch.std(f1_list) * 100)
    print("F1:", f1_list)
    print("Precision:", torch.mean(precision_list) * 100, "+-", torch.std(precision_list) * 100)
    print("Precision:", precision_list)
    print("Recall:", torch.mean(recall_list) * 100, "+-", torch.std(recall_list) * 100)
    print("Recall:", recall_list)
