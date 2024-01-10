"""
-*- coding: utf-8 -*-

@Time    : 2021/4/26 15:16

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : surrogate_act.py
"""
import torch


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

class RectSurrogate(torch.autograd.Function):
    """
    activation function: rectangular function h(*)
    """
    alpha = 0.4
    @staticmethod
    def forward(ctx, input):
        """
           input = vin -thresh
        """
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, delta):
        vin, = ctx.saved_tensors
        # delta_tmp = delta.clone() ####???????
        dgdv = 1.0 / RectSurrogate.alpha * (abs(input) < (RectSurrogate.alpha / 2.0))
        return delta * dgdv

