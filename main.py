from GetData import dvs_ges
import argparse
import os
from torch.utils.data import DataLoader
from Resnet import *
import tqdm
from loss_func import *
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r'/data3/ql/DVSG/hdf5', type=str)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=int)
    parser.add_argument("--C", default=4, type=int)
    parser.add_argument("--C_sim", default=0.97, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--mode", default='ATCA', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--out_dir", default='./checkpoints', type=str)
    parser.add_argument("--vth", default=0.5, type=int)
    parser.add_argument("--net", default='CifarNet', type=str)
    parser.add_argument("--resume", default=True, type=bool)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_ids = [0, 1, 2]
    device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")
    train_data = dvs_ges(args.root, device)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    test_data = dvs_ges(args.root, device, is_train=False)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, num_workers=0)

    if args.net == 'Resnet19':
        snn = resnet19_NoBn()
        print('Resnet')
    else:
        snn = Cifar_Net_NoBn()
        print('CifarNet')


    # loss_func = nn.CrossEntropyLoss()
    if args.mode == 'STCA':
        loss_func = STCA_Last_Loss(args)
    if args.mode == 'ATCA':
        loss_func = Back_Layer_ATCA(args)
    if args.mode == 'Accumulate':
        loss_func = Accumulate_Loss(args)
        snn = Cifar_Net_Accumulate()
        print('Cifar_Net_Accumulate')
    if args.mode == 'Mean':
        loss_func = Mean_Loss(args)
        print('Cifar_Net_Mean')

    snn = nn.DataParallel(snn, device_ids=device_ids)
    snn.to(device)

    if args.resume:
        model = torch.load('./checkpoints/CifarNet_ATCA_3_model.pth')
        print(model['max_test_acc'])
        snn.load_state_dict(model['net'])

    optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    max_test_acc = 0
    loss_trace = []
    acc_trace = []
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for i, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
            snn.train()
            labels = labels.to(device)
            optimizer.zero_grad()
            spike_out, v_trace, v_trace_last = snn(images)
            # out = snn(images)
            loss, _ = loss_func(v_trace, v_trace_last, labels)  # loss, batch*neuron
            # loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_samples += labels.numel()
            train_loss += loss.detach().item()
            # train_acc += (spike_out.sum(dim=2).argmax(dim=1) == labels).float().sum().item()
            if args.mode == 'Accumulate':
                pred = v_trace.argmax(dim=1)
            else:
                pred, _ = v_trace.max(dim=2)
                pred = pred.argmax(dim=1)
            train_acc += (pred == labels).float().sum().item()
            # train_acc += (out.argmax(dim=1).cpu() == labels.cpu()).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples
        lr_scheduler.step()

        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm.tqdm(test_loader)):
                labels = labels.to(device)
                optimizer.zero_grad()
                snn.eval()
                spike_out, v_trace, v_trace_last = snn(images)
                # out = snn(images)
                testloss, _ = loss_func(v_trace, v_trace_last, labels)
                # testloss = loss_func(out, labels)

                test_samples += labels.numel()
                test_loss += testloss.detach().item()
                # test_acc += (spike_out.sum(dim=2).argmax(dim=1) == labels).float().sum().item()
                if args.mode == 'Accumulate':
                    pred = v_trace.argmax(dim=1)
                else:
                    pred, _ = v_trace.max(dim=2)
                    pred = pred.argmax(dim=1)
                test_acc += (pred == labels).float().sum().item()
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
            torch.save(checkpoint, os.path.join(args.out_dir, args.net + '_' + args.mode + '_' + args.seed.__str__() + '_model.pth'))

        print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')

    trace = {
        'loss_trace': loss_trace,
        'acc_trace': acc_trace
    }
    torch.save(trace, os.path.join(args.out_dir, args.net + '_' + args.mode + '_' + args.seed.__str__() + '_trace.pth'))
