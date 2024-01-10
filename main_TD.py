"""
-*- coding: utf-8 -*-

@Time    : 2021/10/21 10:17

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main_TD.py
"""
import scipy.io as io
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.ptn_loader import sPtn, sPtn_TCA
from model.surrogate_act import SurrogateHeaviside
from model.snn import STCADenseLayer, ReadoutLayer, SNN, STCAOutputLayer
from loss_func import *
from params import *


def getSpkLoader(train_path, test_path, train_batch_size=1, test_batch_size=1):
    train_data = io.loadmat(train_path)
    test_data = io.loadmat(test_path)
    train_data_ = sPtn_TCA(train_data['ptnTrain'], train_data['train_labels'], train_data['TmaxTrain'], dt=dt, TCA_slice=TCA_slice)
    test_data_ = sPtn_TCA(test_data['ptnTest'], test_data['test_labels'], test_data['TmaxTest'], dt=dt, TCA_slice=TCA_slice)
    batch_train = torch.utils.data.DataLoader(train_data_, batch_size=train_batch_size, shuffle=False, num_workers=0)
    batch_test = torch.utils.data.DataLoader(test_data_, batch_size=test_batch_size, shuffle=True, num_workers=0)
    return batch_train, batch_test


def trans_labels(batch_size, num_neuron, labels, mode='train'):
    labels_t = torch.zeros(batch_size, num_neuron)
    if mode == 'test':
        for i in range(batch_size):
            labels_t[i][labels[i]] += 1
    else:
        for i in range(batch_size):
            for j in range(TCA_slice):
                labels_t[i][labels[j][i]] += 1
    return labels_t


def runTrain(train_ldr, optimizer, snn, evaluator, ratio):
    global device, dtype
    loss_record = []
    predict_tot = []
    label_tot = []
    snn.train()
    for idx, (ptns, labels) in enumerate(train_ldr):
        ptns = ptns.permute([0, 2, 1]).contiguous().to(device, dtype) # batch*Time*Neuron
        # for i in range(3):
        #     labels[i] = labels[i].to(device)
        optimizer.zero_grad()
        output, vlast, _ = snn(ptns)
        labels_t = trans_labels(output.shape[0], output.shape[2], labels)
        loss, spike_output = evaluator(output, vlast, labels_t, ratio)
        loss.backward()
        torch.nn.utils.clip_grad_value_(snn.parameters(), 5)
        optimizer.step()
        snn.clamp()
        # predict = torch.argmax(spike_output, axis=1).to(device)
        # _, predict = torch.max(torch.max(output, 1)[0], 1)
        loss_record.append(loss.detach().cpu())
        predict_tot.append(spike_output)
        label_tot.append(labels_t)
    predict_tot = torch.cat(predict_tot)
    label_tot = torch.cat(label_tot)
    train_acc = torch.mean((predict_tot == label_tot).float())
    train_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return train_acc, train_loss


def runTest(val_ldr, snn, evaluator):
    global test_trace, device, dtype
    snn.eval()
    with torch.no_grad():
        loss_record = []
        predict_tot = []
        label_tot = []
        for idx, (ptns, labels) in enumerate(val_ldr):
            ptns = ptns.permute([0, 2, 1]).contiguous().to(device, dtype)
            output, vlast, _ = snn(ptns)
            for i in range(TCA_slice):
                labels[i] = labels[i].to(device)
            labels_t = trans_labels(output.shape[0], output.shape[2], labels)
            loss, spike_output = evaluator(output, vlast, labels_t, 0)
            loss_record.append(loss)
            # predict = torch.argmax(spike_output, axis=1).to(device)
            # _, predict = torch.max(torch.max(output, 1)[0], 1)
            predict_tot.append(spike_output)
            label_tot.append(labels_t)
        predict_tot = torch.cat(predict_tot)
        label_tot = torch.cat(label_tot)
        val_acc = torch.mean((predict_tot == label_tot).float())
        val_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return val_acc, val_loss


def main():
    model_list = [name for name in os.listdir(save_dir) if name.startswith(prefix)]
    file = prefix + 'cas_' + str(len(model_list))
    save_path = os.path.join(save_dir, file)
    train_ldr, val_ldr = getSpkLoader(train_path, test_path, train_batch_size=79, test_batch_size=77)
    print('Finish loading HeidelbergDigits from: ', save_path)
    # STCA full connected neural network
    layers = []
    # Surrogate Gradient Functiosurrogate_act.pyn
    # SurrogateHeaviside.sigma = 2
    spike_fn = SurrogateHeaviside.apply
    layers.append(STCADenseLayer(structure[0], structure[1], spike_fn, w_init_mean, w_init_std, recurrent=False,
                                 lateral_connections=False, fc_drop=fc_drop, return_vmem=True))
    # layers.append(SpikingDenseLayer(structure[1],structure[2],spike_fn,w_init_mean,w_init_std,recurrent=False,lateral_connections=True))
    layers.append(STCAOutputLayer(structure[1], structure[2], spike_fn, w_init_mean, w_init_std, recurrent=False, lateral_connections=False))

    snn = SNN(layers).to(device, dtype)
    if pre_train:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(checkpoint['best_acc'])
        snn.load_state_dict(checkpoint['best_net'])
    # optimizer = RAdam(snn.parameters(), lr=3e-4)
    # optimizer = torch.optim.AdamW(snn.parameters(), lr=3e-4, weight_decay=1e-3)
    # optimizer = torch.optim.AdamW(snn.parameters(), lr=3e-4)
    optimizer = torch.optim.Adam(snn.parameters(), lr=lr, amsgrad=True)
    # optimizer = torch.optim.AdamW(snn.parameters(), lr=3e-4, amsgrad=True)
    # evaluator = torch.nn.CrossEntropyLoss()
    evaluator = STCA_TCA_Loss()

    # define some hyperparameter and holk for a run
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    best_acc = 0
    best_train_acc = 0
    test_trace = []
    train_trace = []
    loss_trace = []
    if (not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    start_epoch = 0

    # resume
    if (load_if):
        state = torch.load(os.path.join(save_dir, file + '.t7'))
        snn.load_state_dict(state['best_net'])
        start_epoch = state['best_epoch']
        train_trace = state['traces']['train']
        test_trace = state['traces']['test']
        loss_trace = state['traces']['loss']

    # run  forward - backward
    for epoch in tqdm(range(start_epoch, start_epoch + epoch_num)):
        ratio = epoch / epoch_num
        train_acc, train_loss = runTrain(train_ldr, optimizer, snn, evaluator, ratio)
        train_trace.append(train_acc)
        loss_trace.append(train_loss)
        print('\ntrain record: ', train_loss, train_acc)

        val_acc, val_loss = runTest(val_ldr, snn, evaluator)
        test_trace.append(val_acc)
        print('validation record:', val_loss, val_acc)
        if (val_acc > best_acc):
            best_acc = val_acc
            best_train_acc = train_acc
            print('Saving model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': snn.state_dict(),
                'traces': {'train': train_trace, 'test': test_trace, 'loss': loss_trace},
                # 'raw_pth': ptn_path
            }
            torch.save(state, os.path.join(save_path + '.t7'))
    print("Best Acc:", best_acc)
    print("Best Train Acc:", best_train_acc)


if __name__ == '__main__':
    main()
