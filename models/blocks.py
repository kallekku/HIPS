"""
This module implements some basic blocks used in the ResNet NNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ConvLayerNorm(nn.Module):
    """
    Implements LayerNorm for convolutional layers with associated reshaping
    """
    def __init__(self, norm, elems):
        super().__init__()
        self.norm = norm(elems)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = self.norm(x)
        x = x.view(*x_shape)
        return x


def get_norm(norm, channels, groups=16, momentum=0.0001, elems=128):
    """
    Wrapper for norm
    """
    if norm == nn.GroupNorm:
        return norm(groups, channels)
    elif norm == nn.BatchNorm2d:
        return norm(channels, momentum=momentum)
    elif norm == nn.LayerNorm:
        return ConvLayerNorm(norm, elems)
    else:
        return norm(channels)


class ResNetBlock(nn.Module):
    """
    Building block for ResNet
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 stride=1, norm=nn.BatchNorm2d, groups=16, elems=128, dropout=False):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=False),
            get_norm(norm, out_channels, groups, elems=elems),
            nn.ReLU())
        if dropout:
            self.conv1.append(nn.Dropout(0.1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
            get_norm(norm, out_channels, groups, elems=elems))

        if stride != 1 or out_channels != in_channels:
            self.skip = True
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                get_norm(norm, out_channels, groups, elems=elems))
        else:
            self.skip = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.skip:
            x = self.skip_conv(x)
        y = F.relu(y + x)
        return y


class GroupOfBlocks(nn.Module):
    """
    Group of sequential ResNetBlocks
    """
    def __init__(self, in_channels, out_channels, n_blocks, stride=1,
                 norm=nn.BatchNorm2d, elems=None, dropout=False):
        super().__init__()
        first_block = ResNetBlock(in_channels, out_channels, stride=stride,
                                  norm=norm, elems=elems, dropout=dropout)
        other_blocks = [ResNetBlock(out_channels, out_channels, norm=norm,
                                    elems=elems, dropout=dropout)
                        for _ in range(1, n_blocks)]
        self.group = nn.Sequential(first_block, *other_blocks)

    def forward(self, x):
        return self.group(x)


class FilmResBlock(nn.Module):
    """
    ResNet Blocks with FiLM layers
    """
    def __init__(self, in_channels, out_channels, fh, padding=1, stride=1,
                 kernel_size=3, norm=nn.BatchNorm2d):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn = norm(out_channels)
        self.lin1 = nn.Linear(fh, out_channels*2)

        if stride != 1 or out_channels != in_channels:
            self.skip = True
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                norm(out_channels),
            )
        else:
            self.skip = False

    def forward(self, x, z):
        x = F.relu(self.conv1(x))
        y = self.bn(self.conv2(x))

        z = self.lin1(z)
        gamma = z[:, :self.out_channels].unsqueeze(2).unsqueeze(2)
        beta = z[:, self.out_channels:].unsqueeze(2).unsqueeze(2)

        y = gamma*y + beta

        if hasattr(self, 'skip') and self.skip:
            x = self.skip_conv(x)

        x = x + F.relu(y)
        return x


# From https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
class VectorQuantization(Function):
    """
    Implements vector quantization
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook**2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten,
                                    codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError


class VectorQuantizationStraightThrough(Function):
    """
    Adds backward-function with straight-through gradient to VectorQuantization.
    This class depends on the class VectorQuantization
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()

        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)
            grad_output_flatten = (grad_output.contiguous().
                                   view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
