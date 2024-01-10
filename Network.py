from layer import *


class FCnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, mode='Fix'):
        super(FCnet, self).__init__()
        self.l1 = tdLayer(nn.Linear(input_dim, hidden_dim), device)
        self.l2 = tdLayer(nn.Linear(hidden_dim, output_dim), device)
        self.mode = mode
        if mode == 'Fix':
            self.spike = C_LIF_Fix()
        elif mode == 'Share':
            self.spike = C_LIF_Share()
        elif mode == 'Layer':
            self.spike1 = C_LIF_Share()
            self.spike2 = C_LIF_Share()
        elif mode == 'Indenpent':
            self.spike1 = C_LIF(hidden_dim)
            self.spike2 = C_LIF(output_dim)

    def forward(self, x):
        if self.mode != 'Layer' and self.mode != 'Indenpent':
            x = self.spike(self.l1(x.float()))
            x = self.spike(self.l2(x), True, False)
        else:
            x = self.spike1(self.l1(x.float()))
            x = self.spike2(self.l2(x), True, False)
        return x


if __name__ == '__main__':
    x = torch.ones((50, 384, 151), dtype=torch.float32)
    snn = FCnet(384, 1024, 10)
    print(snn(x).shape)
