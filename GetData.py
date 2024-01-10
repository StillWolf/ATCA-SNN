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
