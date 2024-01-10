import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spikingjelly.datasets import play_frame
import scipy.io as io


class dvs_ges(Dataset):
    def __init__(self, record_path, device, is_train=True):
        self.data = []
        self.is_train = is_train
        self.channel = 2
        self.size = 64
        self.device = device
        for i in range(11):
            if is_train:
                data_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_train_image_dve_ges.hdf5'
                label_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_train_label_dve_ges.hdf5'
            else:
                data_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_test_image_dve_ges.hdf5'
                label_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_test_label_dve_ges.hdf5'
            data_frame = np.array(h5py.File(data_path, 'r')['data'])
            label_frame = np.array(h5py.File(label_path, 'r')['label'])
            length = data_frame.shape[0]

            for j in range(length):
                self.data.append([data_frame[j,:], label_frame[j]])
        self.transformations = transforms.Compose([transforms.ToTensor()])

    # 获取单条数据
    def __getitem__(self, index):
        data = self.data[index][0]
        img = torch.tensor(data.reshape(2, 32, 32, 40), dtype=torch.float32, device=self.device)
        label = int(self.data[index][1])
        return img, label

    # 数据集长度
    def __len__(self):
        return len(self.data)


class dvs_ges_tca(Dataset):
    def __init__(self, record_path, device, slice, is_train=True):
        self.data = []
        self.is_train = is_train
        self.channel = 2
        self.size = 64
        self.device = device
        self.slice = slice
        for i in range(11):
            if is_train:
                data_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_train_image_dve_ges.hdf5'
                label_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_train_label_dve_ges.hdf5'
            else:
                data_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_test_image_dve_ges.hdf5'
                label_path = record_path + '/' + str(i) + '_csv' + '/' + str(i) + '_test_label_dve_ges.hdf5'
            data_frame = np.array(h5py.File(data_path, 'r')['data'])
            label_frame = np.array(h5py.File(label_path, 'r')['label'])
            length = data_frame.shape[0]

            for j in range(length):
                self.data.append([data_frame[j, :], label_frame[j]])
        self.transformations = transforms.Compose([transforms.ToTensor()])

    # 获取单条数据
    def __getitem__(self, index):
        label = []
        noise = torch.zeros(2, 32, 32, 10).to(self.device)
        for i in range(self.slice):
            index_tca = np.random.randint(len(self.data))
            while self.data[index_tca][1] in label:
                index_tca = np.random.randint(len(self.data))
            data = self.data[index_tca][0]
            label.append(self.data[index_tca][1])
            img = torch.tensor(data.reshape(2, 32, 32, 40), dtype=torch.float32, device=self.device)
            img = torch.cat((img, noise), 3)
            if i == 0:
                img_tca = img
            else:
                img_tca = torch.cat((img_tca, img), 3)
        label = torch.tensor(label, dtype=torch.int64)
        return img_tca, label

    # 数据集长度
    def __len__(self):
        return int(len(self.data)/self.slice)


class sPtn(Dataset):
    def __init__(self, dataSource, labelSource, TmaxSource, mode='spike', dt=None):
        self.mode = mode
        if mode == 'spike':
            """
            dataSource: num_samples x max_num_spikes x 2, pad with -1 for inconsistency in spiking counts.
                        eg:
                            num_samples = 2 
                            max_num_spikes = 5
                            sample 1:
                                for neuron 1 , the afferent is [2, 5, 7]
                                for neuron 2 , the afferent is [1, 9]
                            sample 2: 
                                for neuron 1 : the afferent is [2, 9, 14]
                                for neuron 2 , the afferent is [6]
                            ==============================================================================
                            then dataSource = [
                                                [[1,2],[1,5],[1,7],[2,1],[2,9]], # for sample 1 
                                                [[1,2],[1,9],[1,14],[2,6],[-1,-1]] # for sample 2
                                                                ]]
            """
            self.dataSource = dataSource
            self.labelSource = labelSource
            self.TmaxSource = TmaxSource
        elif mode == 'neuron':
            assert not dt is None, 'for neuron-based format, dt should not be None for iterative SNNs'
            self.dt = dt
            self.nb_steps = int(np.ceil(TmaxSource.max() / dt)) + 1
            self.dataSource = dataSource
            self.labelSource = labelSource
            self.TmaxSource = TmaxSource
        elif mode == 'time':
            raise NotImplementedError('Time-based loading is not supported util now.')
        else:
            raise NotImplementedError("Only 'spike' ,'neuron' and 'time' format are in the plan.")

    def __getitem__(self, index):
        if self.mode == 'spike':
            spike = self.dataSource[index, :, :]
            spike = np.transpose(spike)
            label = self.labelSource[index]
            Tmax = float(self.TmaxSource[index])
            return spike, Tmax, label
        elif self.mode == 'neuron':
            data = self.dataSource[index]
            ptn = np.zeros([len(data), self.nb_steps])
            for i, aff in enumerate(data):
                aff = np.array(aff).reshape([-1])
                if len(aff) > 0:
                    ptn[np.ones_like(aff, dtype=np.int) * i, np.round(aff / self.dt).astype(np.int)] = 1
            Tmax = float(self.TmaxSource[index])
            label = int(self.labelSource[index]) - 1
            return ptn, label

    def __len__(self):
        return self.dataSource.shape[0]


class sPtn_MDB10(Dataset):
    def __init__(self, dataSource, labelSource, TmaxSource, mode='spike', dt=None):
        self.mode = mode
        if mode == 'spike':
            """
            dataSource: num_samples x max_num_spikes x 2, pad with -1 for inconsistency in spiking counts.
                        eg:
                            num_samples = 2 
                            max_num_spikes = 5
                            sample 1:
                                for neuron 1 , the afferent is [2, 5, 7]
                                for neuron 2 , the afferent is [1, 9]
                            sample 2: 
                                for neuron 1 : the afferent is [2, 9, 14]
                                for neuron 2 , the afferent is [6]
                            ==============================================================================
                            then dataSource = [
                                                [[1,2],[1,5],[1,7],[2,1],[2,9]], # for sample 1 
                                                [[1,2],[1,9],[1,14],[2,6],[-1,-1]] # for sample 2
                                                                ]]
            """
            self.dataSource = dataSource
            self.labelSource = labelSource
            self.TmaxSource = TmaxSource
        elif mode == 'neuron':
            assert not dt is None, 'for neuron-based format, dt should not be None for iterative SNNs'
            self.dt = dt
            # print(self.nb_steps)
            self.dataSource = dataSource[0][0]
            self.labelSource = labelSource.T[0][0][0]
            self.TmaxSource = TmaxSource[0][0]
            self.nb_steps = int(np.ceil(self.TmaxSource.max() / dt)) + 1
        elif mode == 'time':
            raise NotImplementedError('Time-based loading is not supported util now.')
        else:
            raise NotImplementedError("Only 'spike' ,'neuron' and 'time' format are in the plan.")

    def __getitem__(self, index):
        if self.mode == 'spike':
            spike = self.dataSource[index, :, :]
            spike = np.transpose(spike)
            label = self.labelSource[index]
            Tmax = float(self.TmaxSource[index])
            return spike, Tmax, label
        elif self.mode == 'neuron':
            data = self.dataSource[index]
            ptn = np.zeros([len(data), self.nb_steps])
            for i, aff in enumerate(data):
                aff = np.array(aff).reshape([-1])
                # print(aff)
                if len(aff) > 0:
                    ptn[np.ones_like(aff, dtype=np.int) * i, np.round(aff / self.dt).astype(np.int)] = 1
            label = int(self.labelSource[index])
            return ptn, label - 1

    def __len__(self):
        return self.dataSource.shape[0]


def getSpkLoader(train_path, test_path, train_batch_size=1, test_batch_size=1):
    train_data = io.loadmat(train_path)
    test_data = io.loadmat(test_path)
    train_data_ = sPtn(train_data['ptnTrain'], train_data['train_labels'], train_data['TmaxTrain'], mode='neuron',
                       dt=1e-3)
    test_data_ = sPtn(test_data['ptnTest'], test_data['test_labels'], test_data['TmaxTest'], mode='neuron', dt=1e-3)
    # func = collate_func(mode, num_in)
    batch_train = torch.utils.data.DataLoader(train_data_, batch_size=train_batch_size, shuffle=True, num_workers=0)
    batch_test = torch.utils.data.DataLoader(test_data_, batch_size=test_batch_size, shuffle=False, num_workers=0)
    return batch_train, batch_test


def getSpkLoader_for_MDB10(data_path, train_batch_size=1, test_batch_size=1):
    data = io.loadmat(data_path)
    train_data_ = sPtn_MDB10(data['TrainData']['ptn'], data['TrainData']['Labels'], data['TrainData']['Tmax'], mode='neuron', dt=0.01)
    test_data_ = sPtn_MDB10(data['TestData']['ptn'], data['TestData']['Labels'], data['TestData']['Tmax'], mode='neuron', dt=0.01)
    batch_train = torch.utils.data.DataLoader(train_data_, batch_size=train_batch_size, shuffle=True, num_workers=0)
    batch_test = torch.utils.data.DataLoader(test_data_, batch_size=test_batch_size, shuffle=False, num_workers=0)
    return batch_train, batch_test


if __name__ == '__main__':
    path = r'/data3/ql/DVSG/hdf5'
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = dvs_ges(path, device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_data = dvs_ges(path, device, is_train=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
        print(labels)
        play_frame(images[0].permute(3, 0, 1, 2), save_gif_to='./gif/'+i.__str__()+'.gif')
