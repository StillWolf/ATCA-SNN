import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear(torch.autograd.Function):
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient
    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = Linear.gamma * F.threshold(1.0 - torch.abs(inpt), 0, 0)
        return grad_input * sur_grad.float()


class Rectangle(torch.autograd.Function):
    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = (torch.abs(inpt) < 0.5).float()
        return grad_input * sur_grad


class PDF(torch.autograd.Function):

    alpha = 0.1
    beta = 0.1

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = PDF.alpha * torch.exp(-PDF.beta * torch.abs(inpt))
        return sur_grad * grad_input


class SurrogateHeaviside(torch.autograd.Function):
    # Activation function with surrogate gradient
    sigma = 10.0

    @staticmethod
    def forward(ctx, input):
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = grad_input * torch.sigmoid(SurrogateHeaviside.sigma * input) * torch.sigmoid(
            -SurrogateHeaviside.sigma * input)
        return grad


Act = SurrogateHeaviside.apply


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=30):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * init_lr
    return optimizer


def state_update(u, o, i, decay, Vth):
    u = decay * u + i - o * Vth
    o = Act(u - Vth)
    return u, o


def CLIF_update(M, S, E, o, i, decay_m, decay_s, Vth):
    M = decay_m * (M + i)
    S = decay_s * (S + i)
    E = decay_m * E + o * Vth.detach()
    u = M-S-E-Vth
    o = Act(u)
    return M, S, E, o


def CLIF_accumulated_state(M, S, i, decay_m, decay_s):
    M = decay_m * (M + i)
    S = decay_s * (S + i)
    return M, S


def accumulated_state(u, o):
    u_ = 0.5 * u + o
    return u_


class LIF(nn.Module):
    def __init__(self):
        super(LIF, self).__init__()
        print("LIF")
        init_decay = 0.5
        ini_v = 0.5

        #self.nrom = torch.norm(w.detach().cpu(), None, dim=None)
        self.decay = nn.Parameter(torch.tensor(init_decay, dtype=torch.float), requires_grad=False)
        self.decay.data.clamp_(0., 1.)
        self.vth = ini_v

    def forward(self, x, output=False, vmem=False):
        steps = x.shape[-1]
        if output:
            if not vmem:
                u = torch.zeros(x.shape[:-1], device=x.device)
                for step in range(steps):
                    u = accumulated_state(u, x[..., step])
                return u
            else:
                u = torch.zeros(x.shape[:-1], device=x.device)
                out = torch.zeros(x.shape, device=x.device)
                u_trace = torch.zeros(x.shape, device=x.device)
                for step in range(steps):
                    u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step], self.decay, self.vth)
                    u_trace[..., step] = u
                return out, u_trace - self.vth

        else:
            u = torch.zeros(x.shape[:-1], device=x.device)
            out = torch.zeros(x.shape, device=x.device)
            for step in range(steps):
                u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step], self.decay, self.vth)
            return out


class C_LIF(nn.Module):
    def __init__(self, shape):
        super(C_LIF, self).__init__()
        print("C_LIF")
        init_decay_m = 0.9
        init_decay_s = 0.6
        ini_v = 1.0

        #self.nrom = torch.norm(w.detach().cpu(), None, dim=None)
        self.decay_m = nn.Parameter(torch.empty(shape, dtype=torch.float), requires_grad=True)
        self.decay_s = nn.Parameter(torch.empty(shape, dtype=torch.float), requires_grad=True)
        torch.nn.init.normal_(self.decay_m, mean=init_decay_m, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=init_decay_s, std=0.01)
        self.decay_m.data.clamp_(0., 1.)
        self.decay_s.data.clamp_(0., 1.)
        self.vth = nn.Parameter(torch.empty(shape, dtype=torch.float), requires_grad=True)
        torch.nn.init.normal_(self.vth, mean=ini_v, std=0.01)
        self.vth.data.clamp_(0., 1.)

    def forward(self, x, output=False, vmem=False):
        steps = x.shape[-1]
        M = torch.zeros(x.shape[:-1], device=x.device)
        S = torch.zeros(x.shape[:-1], device=x.device)
        E = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        u_trace = torch.zeros(x.shape, device=x.device)
        if output and not vmem:
            for step in range(steps):
                M, S = CLIF_accumulated_state(M, S, x[..., step], self.decay_m, self.decay_s)
            return M-S
        for step in range(steps):
            M, S, E, out[..., step] = CLIF_update(M, S, E, out[..., max(step - 1, 0)], x[..., step], self.decay_m, self.decay_s, self.vth)
            u_trace[..., step] = M-S-E
        if not output:
            return out
        else:
            return out, u_trace - self.vth


class C_LIF_Share(nn.Module):
    def __init__(self):
        super(C_LIF_Share, self).__init__()
        print("C_LIF_Share")
        init_decay_m = 0.9
        init_decay_s = 0.6
        ini_v = 1.0

        #self.nrom = torch.norm(w.detach().cpu(), None, dim=None)
        self.decay_m = nn.Parameter(torch.empty(1, dtype=torch.float), requires_grad=True)
        self.decay_s = nn.Parameter(torch.empty(1, dtype=torch.float), requires_grad=True)
        self.vth = nn.Parameter(torch.empty(1, dtype=torch.float), requires_grad=True)
        torch.nn.init.normal_(self.decay_m, mean=init_decay_m, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=init_decay_s, std=0.01)
        torch.nn.init.normal_(self.vth, mean=ini_v, std=0.01)
        self.decay_m.data.clamp_(0., 1.)
        self.decay_s.data.clamp_(0., 1.)
        self.vth.data.clamp_(0., 1.)
        # self.vth = nn.Parameter(torch.empty(shape, dtype=torch.float), requires_grad=True)
        # torch.nn.init.normal_(self.vth, mean=self.ini_v, std=0.01)

    def forward(self, x, output=False, vmem=False):
        steps = x.shape[-1]
        M = torch.zeros(x.shape[:-1], device=x.device)
        S = torch.zeros(x.shape[:-1], device=x.device)
        E = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        u_trace = torch.zeros(x.shape, device=x.device)
        if output and not vmem:
            for step in range(steps):
                M, S = CLIF_accumulated_state(M, S, x[..., step], self.decay_m, self.decay_s)
            return M-S
        for step in range(steps):
            M, S, E, out[..., step] = CLIF_update(M, S, E, out[..., max(step - 1, 0)], x[..., step], self.decay_m, self.decay_s, self.vth)
            u_trace[..., step] = M-S-E
        if not output:
            return out
        else:
            return out, u_trace - self.vth


class C_LIF_Fix(nn.Module):
    def __init__(self):
        super(C_LIF_Fix, self).__init__()
        print("C_LIF_Fix")
        init_decay_m = 0.9
        init_decay_s = 0.6
        ini_v = 1

        self.decay_m = init_decay_m
        self.decay_s = init_decay_s
        self.vth = ini_v

    def forward(self, x, output=False, vmem=False):
        steps = x.shape[-1]
        M = torch.zeros(x.shape[:-1], device=x.device)
        S = torch.zeros(x.shape[:-1], device=x.device)
        E = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        u_trace = torch.zeros(x.shape, device=x.device)
        if output and not vmem:
            for step in range(steps):
                M, S = CLIF_accumulated_state(M, S, x[..., step], self.decay_m, self.decay_s)
            return M-S
        for step in range(steps):
            M, S, E, out[..., step] = CLIF_update(M, S, E, out[..., max(step - 1, 0)], x[..., step], self.decay_m, self.decay_s, self.vth)
            u_trace[..., step] = M-S-E
        if not output:
            return out
        else:
            return out, u_trace - self.vth


class tdLayer(nn.Module):
    def __init__(self, layer, device):
        super(tdLayer, self).__init__()
        self.device = device
        self.layer = layer.to(device)

    def forward(self, x):
        # print(x[..., 0].device)
        T = x.shape[-1]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (T,), device=self.device)
        for step in range(T):
            x_[..., step] = self.layer(x[..., step])
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, bn, alpha=1, Vth = 0.5):
        super(tdBatchNorm, self).__init__()
        self.bn = bn
        self.alpha = alpha
        self.Vth = Vth

    def forward(self, x):
        exponential_average_factor = 0.0

        if self.training and self.bn.track_running_stats:
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked += 1
                if self.bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.bn.momentum

        if self.training:
            mean = x.mean([0, 2, 3, 4], keepdim=True)
            var = x.var([0, 2, 3, 4], keepdim=True, unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0, 0]\
                                       + (1 - exponential_average_factor) * self.bn.running_mean
                self.bn.running_var = exponential_average_factor * var[0, :, 0, 0, 0] * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.bn.running_var
        else:
            mean = self.bn.running_mean[None, :, None, None, None]
            var = self.bn.running_var[None, :, None, None, None]

        x = self.alpha * self.Vth * (x - mean) / (torch.sqrt(var) + self.bn.eps)

        if self.bn.affine:
            x = x * self.bn.weight[None, :, None, None, None] + self.bn.bias[None, :, None, None, None]

        return x


class TemporalBN(nn.Module):
    def __init__(self, in_channels, nb_steps):
        super(TemporalBN, self).__init__()
        self.nb_steps = nb_steps
        self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels) for t in range(self.nb_steps)])

    def forward(self, x):
        out = []
        stack_dim = len(x.shape) - 1
        for t in range(self.nb_steps):
            out.append(self.bns[t](x[..., t]))
        out = torch.stack(out, dim=stack_dim)
        return out


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forwad(self, x):
        max = torch.max(x, dim=1, keepdim=True)
        min = torch.min(x, dim=1, keepdim=True)

        x = (max - x)/ (x - min)
        return x
