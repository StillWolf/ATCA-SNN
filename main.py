import torch
from GetData import *
import argparse
import os
from torch.utils.data import DataLoader
from Network import *
from tqdm import tqdm
from loss_func import *
import numpy as np
import time
from sklearn.metrics import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MedleyDB', type=str)
    parser.add_argument("--lr", default=6e-4, type=int)
    parser.add_argument("--C", default=4, type=int)
    parser.add_argument("--C_sim", default=0.97, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--mode", default='TIDIGITS', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--out_dir", default='./checkpoints', type=str)
    parser.add_argument("--vth", default=1, type=int)
    parser.add_argument("--resume", default=False, type=bool)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("set seed:", args.seed)

    device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.dataset == 'TIDIGITS':
        train_ldr, val_ldr = getSpkLoader('./dataset/TIDIGITS/trainSet.mat', './dataset/TIDIGITS/testSet.mat',
                                          train_batch_size=50, test_batch_size=77)
        snn = FCnet(576, 2000, 11, device, mode='Indenpent')
    if args.dataset == 'MedleyDB':
        train_ldr, val_ldr = getSpkLoader_for_MDB10('./dataset/MedleyDB/M10_Data_rank3.mat', train_batch_size=50,
                                                    test_batch_size=50)
        snn = FCnet(384, 1024, 10, device, mode='Indenpent')
    # snn = nn.DataParallel(snn, device_ids=device_ids)
    snn = snn.to(device)

    optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    loss_func = nn.CrossEntropyLoss().to(device)
    max_test_acc = 0
    loss_trace = []
    acc_trace = []
    pred_tot = []
    label_tot = []
    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for idx, (ptns, labels) in enumerate(train_ldr):
            snn.train()
            labels = labels.to(device)
            ptns = ptns.to(device)
            # print(ptns.device)
            optimizer.zero_grad()
            out = snn(ptns)
            loss = loss_func(out, labels)  # loss, batch*neuron
            # loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_samples += labels.numel()
            train_loss += loss.detach().item()
            # train_acc += (spike_out.sum(dim=2).argmax(dim=1) == labels).float().sum().item()
            pred = out.argmax(dim=1)
            train_acc += (pred == labels).float().sum().item()
            # train_acc += (out.argmax(dim=1).cpu() == labels.cpu()).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples
        lr_scheduler.step()

        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for idx, (ptns, labels) in enumerate(val_ldr):
                labels = labels.to(device)
                optimizer.zero_grad()
                snn.eval()
                ptns = ptns.to(device)
                out = snn(ptns)
                # out = snn(images)
                # testloss, _ = loss_func(v_trace, v_trace_last, labels)
                testloss = loss_func(out, labels)

                test_samples += labels.numel()
                test_loss += testloss.detach().item()
                # test_acc += (spike_out.sum(dim=2).argmax(dim=1) == labels).float().sum().item()
                pred = out.argmax(dim=1)
                test_acc += (pred == labels).float().sum().item()
                if idx == 0:
                    labels_tot = labels.cpu()
                    pred_tot = pred.cpu()
                else:
                    labels_tot = torch.cat((labels_tot, labels.cpu()))
                    pred_tot = torch.cat((pred_tot, pred.cpu()))
                # test_acc += (out.argmax(dim=1).cpu() == labels.cpu()).float().sum().item()

        test_loss /= test_samples
        test_acc /= test_samples
        save_max = False

        loss_trace.append(test_loss)
        acc_trace.append(test_acc)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': snn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }
        if save_max:
            torch.save(checkpoint, os.path.join(args.out_dir, args.mode + '_' + args.seed.__str__() + '_model.pth'))

        print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')
        f1 = round(f1_score(labels_tot, pred_tot, average='weighted'), 4)
        precision = round(precision_score(labels_tot, pred_tot, average='macro'), 4)
        recall = round(recall_score(labels_tot, pred_tot, average='macro'), 4)
        print('Precison: {0},\n F1: {1},\n Recall: {2}\n'.format(precision, f1, recall))

    trace = {
        'loss_trace': loss_trace,
        'acc_trace': acc_trace
    }
    torch.save(trace, os.path.join(args.out_dir, args.dataset + '_' + args.seed.__str__() + '_trace.pth'))
