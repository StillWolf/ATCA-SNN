"""
-*- coding: utf-8 -*-

@Time    : 2021/10/21 10:03

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : snn.py

Reference:
    1. Gu, Pengjie, et al. "STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks." IJCAI. 2019.
    2. Zimmer, Romain, et al. "Technical report: supervised training of convolutional spiking neural networks with PyTorch." arXiv preprint arXiv:1911.10124 (2019).
    3. 加速自定义RNN Cell: https://github.com/pytorch/pytorch/blob/963f7629b591dc9750476faf1513bc7f1fb4d6de/benchmarks/fastrnns/custom_lstms.py#L246
"""
import torch
import numpy as np
from .function import td_Dropout


class SNN(torch.nn.Module):

    def __init__(self, layers):

        super(SNN, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        loss_seq = []
        num_layers = len(self.layers)
        for i, l in enumerate(self.layers):
            if i == num_layers - 2:
                x, vmem, loss = l(x)
            else:
                x, loss = l(x)
            loss_seq.append(loss)

        return x, vmem, loss_seq

    def clamp(self):

        for l in self.layers:
                l.clamp()

    def reset_parameters(self):

        for l in self.layers:
                l.reset_parameters()


class STCADenseLayer(torch.nn.Module):

    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean, w_init_std,
                 recurrent=False, lateral_connections=False, rc_drop=-1, fc_drop=-1, eps=1e-8, return_vmem=False):

        super(STCADenseLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_shape, output_shape)), requires_grad=True)

        # RNN Dropout
        self.fc_drop = fc_drop
        self.rc_drop = rc_drop
        if fc_drop > 0:
            self.drop_fc = td_Dropout(fc_drop, True)
        if rc_drop > 0:
            self.drop_rc = td_Dropout(rc_drop, True)

        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)  # threshhold

        # decay
        self.decay_m = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.decay_s = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.decay_e = torch.nn.Parameter(torch.empty(1), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True
        self.return_vmem = return_vmem

    def forward(self, x):

        batch_size = x.shape[0]
        # update drop out mask
        if self.fc_drop > 0:
            self.drop_fc.reset()
        if self.rc_drop > 0:
            self.drop_rc.reset()
        # todo: 爱因斯坦求和标记，支持torch BP
        h = torch.einsum("abc,cd->abd", x, self.w)
        nb_steps = h.shape[1]

        # output spikes
        spk = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        # refractory period kernel
        E = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        # input response kernel
        M = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        S = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)
        vmem = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)
        if self.lateral_connections:  # 计算侧抑制权重
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)  # 对每个输出神经元计算norm以约束权重

        for t in range(nb_steps):
            # input term
            input_ = h[:, t, :]
            if self.recurrent:
                if self.rc_drop > 0:
                    input_ = input_ + torch.einsum("ab,bc->ac", self.drop_rc(spk), self.v)
                else:
                    input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            # todo: add reset mechanism
            if self.lateral_connections:
                # 模拟侧抑制
                E = torch.einsum("ab,bc ->ac", spk, d)
            else:
                # 模拟refractory period
                E = self.decay_m * norm * (E + spk)

            M = self.decay_m * (M + input_)
            S = self.decay_s * (S + input_)

            # membrane potential update
            mem = M - S - E
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b
            if self.return_vmem:
                vmem[:, t, :] = mthr
            spk = self.spike_fn(mthr)
            spk_rec[:, t, :] = spk
            if self.fc_drop > 0:
                spk = self.drop_fc(spk)

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()  # 记录该层的脉冲发放 shape: batch_size x T x NeuIn

        loss = 0.5 * (spk_rec ** 2).mean()
        if self.return_vmem:
            return spk_rec, vmem, loss
        else:
            return spk_rec, loss

    # todo: 将权重的初始化逻辑从module中独立出来形成Initilizer
    def reset_parameters(self):

        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.output_shape))
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_parameters_v2(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.orthogonal(self.v)
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_parameters_v3(self):
        torch.nn.init.orthogonal(self.w)
        if self.recurrent:
            torch.nn.init.orthogonal(self.v)
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_parameters_v4(self):
        torch.nn.init.xavier_normal_(self.w)
        if self.recurrent:
            torch.nn.init.xavier_normal_(self.v)
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    # todo: 将decay改为可学习的softmax门控机制
    def clamp(self):
        self.decay_m.data.clamp_(0., 1.)
        self.decay_e.data.clamp_(0., 1.)
        self.decay_s.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)


class ReadoutLayer(torch.nn.Module):
    "Fully connected readout"

    def __init__(self, input_shape, output_shape, w_init_mean, w_init_std, eps=1e-8, time_reduction="mean"):

        assert time_reduction in ["mean", "max", "volt"], 'time_reduction should be "mean", "max" or "volt"'

        super(ReadoutLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.eps = eps
        self.time_reduction = time_reduction

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if time_reduction == "max" or time_reduction == "volt":
            self.beta = torch.nn.Parameter(torch.tensor(0.7 * np.ones((1))), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.mem_rec_hist = None

    def forward(self, x):

        batch_size = x.shape[0]

        h = torch.einsum("abc,cd->abd", x, self.w)

        norm = (self.w ** 2).sum(0)

        if self.time_reduction == "max" or self.time_reduction == "volt":
            nb_steps = x.shape[1]
            # membrane potential
            mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

            # memrane potential recording
            mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

            for t in range(nb_steps):
                # membrane potential update
                mem = mem * self.beta + (1 - self.beta) * h[:, t, :]
                mem_rec[:, t, :] = mem
            if self.time_reduction == "max":
                output = torch.max(mem_rec, 1)[0] / (norm + 1e-8) - self.b
            else:
                output = mem_rec - self.b  # batch_size * time_window * num_output

        elif self.time_reduction == "mean":

            mem_rec = h
            output = torch.mean(mem_rec, 1) / (norm + 1e-8) - self.b


        # save mem_rec for plotting
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()

        loss = None

        return output, loss

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / (self.input_shape)))

        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)

        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_parameters_v4(self):
        torch.nn.init.xavier_normal_(self.w)

        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)

        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):

        if self.time_reduction == "max":
            self.beta.data.clamp_(0., 1.)


class STCAOutputLayer(torch.nn.Module):

    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean, w_init_std,
                 recurrent=False, lateral_connections=False, rc_drop=-1, fc_drop=-1, eps=1e-8):

        super(STCAOutputLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_shape, output_shape)), requires_grad=True)

        # RNN Dropout
        self.fc_drop = fc_drop
        self.rc_drop = rc_drop
        if fc_drop > 0:
            self.drop_fc = td_Dropout(fc_drop, True)
        if rc_drop > 0:
            self.drop_rc = td_Dropout(rc_drop, True)

        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)  # threshhold

        # decay
        self.decay_m = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.decay_s = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.decay_e = torch.nn.Parameter(torch.empty(1), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]
        # update drop out mask
        if self.fc_drop > 0:
            self.drop_fc.reset()
        if self.rc_drop > 0:
            self.drop_rc.reset()
        # todo: 爱因斯坦求和标记，支持torch BP
        h = torch.einsum("abc,cd->abd", x, self.w)
        nb_steps = h.shape[1]

        # output spikes
        spk = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        # refractory period kernel
        E = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        # input response kernel
        M = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        S = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

        if self.lateral_connections:  # 计算侧抑制权重
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)  # 对每个输出神经元计算norm以约束权重

        for t in range(nb_steps):
            # input term
            input_ = h[:, t, :]
            if self.recurrent:
                if self.rc_drop > 0:
                    input_ = input_ + torch.einsum("ab,bc->ac", self.drop_rc(spk), self.v)
                else:
                    input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            # todo: add reset mechanism
            if self.lateral_connections:
                # 模拟侧抑制
                E = torch.einsum("ab,bc ->ac", spk, d)
            else:
                # 模拟refractory period
                E = self.decay_m * norm * (E + spk)

            M = self.decay_m * (M + input_)
            S = self.decay_s * (S + input_)

            # membrane potential update
            mem = M - S - E
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b

            spk_rec[:, t, :] = mthr
            if self.fc_drop > 0:
                spk = self.drop_fc(spk)

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()  # 记录该层的脉冲发放 shape: batch_size x T x NeuIn

        loss = 0.5 * (spk_rec ** 2).mean()

        return spk_rec, loss

    # todo: 将权重的初始化逻辑从module中独立出来形成Initilizer
    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.output_shape))
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    # todo: 将decay改为可学习的softmax门控机制
    def clamp(self):
        self.decay_m.data.clamp_(0., 1.)
        self.decay_e.data.clamp_(0., 1.)
        self.decay_s.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)