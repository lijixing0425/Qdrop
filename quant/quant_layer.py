import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
def calculate_channel_entropy(feature_map, bins=256):
    batch_size, num_channels, _, _ = feature_map.shape

    # Normalize the feature map to [0, 1]
    min_val = feature_map.view(batch_size, num_channels, -1).min(dim=2)[0].view(batch_size, num_channels, 1, 1)
    max_val = feature_map.view(batch_size, num_channels, -1).max(dim=2)[0].view(batch_size, num_channels, 1, 1)
    normalized = (feature_map - min_val) / (max_val - min_val + 1e-5)  # Adding a small value to avoid division by zero

    # Initialize entropy tensor
    entropy = torch.zeros(num_channels)

    # Calculate entropy for each channel
    for i in range(num_channels):
        channel_data = normalized[:, i, :, :].flatten().cpu().numpy()
        hist, bin_edges = np.histogram(channel_data, bins=bins, density=True)
        hist = hist + 1e-5  # Adding a small value to avoid log of zero
        hist = hist / hist.sum()  # Normalize the histogram to form a probability distribution

        # Calculate entropy
        entropy[i] = -np.sum(hist * np.log2(hist))

    return entropy.cuda()

def pso_asymmetric_quantization_tensor(weight, num_bits, num_particles, iterations):
    qmin = 0
    qmax = 2 ** num_bits - 1

    # Reshape weights for per-channel processing
    reshaped_weights = weight.view(-1)

    # Initialize particle positions and velocities
    L = torch.max(reshaped_weights) - torch.min(reshaped_weights)
    position = torch.randn((num_particles, 2), device=weight.device)
    position[:, 0] = torch.max(reshaped_weights) * torch.rand((num_particles), device=weight.device)
    position[:, 0] = torch.maximum(position[:, 0], torch.max(reshaped_weights) - 0.5 * L)
    position[:, 1] = torch.min(reshaped_weights) * torch.rand((num_particles), device=weight.device)
    position[:, 1] = torch.minimum(position[:, 1], torch.min(reshaped_weights) + 0.5 * L)
    velocity = torch.zeros((num_particles, 2), device=weight.device)

    # Personal best positions and values
    personal_best_position = position.clone()
    personal_best_value = torch.full((num_particles, 1), float('inf'), device=weight.device)

    # Global best positions and values
    global_best_position = torch.zeros((1, 2), device=weight.device)
    global_best_value = torch.full((1, 1), float('inf'), device=weight.device)

    # PSO constants
    c1, c2 = 1.5, 1.5  # cognitive and social constants
    w = 0.5  # inertia weight

    for i in range(iterations):
        # Calculate quantized weights and MSE
        scale = (position[:, 0].unsqueeze(-1) - position[:, 1].unsqueeze(-1)) / (qmax)
        zero_point = -torch.round(position[:, 1].unsqueeze(-1) / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        quantized = (torch.clamp(torch.round(reshaped_weights.unsqueeze(0) / scale) + zero_point, qmin, qmax) - zero_point) * scale + zero_point
        mse = torch.mean((reshaped_weights.unsqueeze(0) - quantized) ** 2, dim=-1, keepdim=True)

        # Update personal best
        personal_best_mask = mse < personal_best_value
        personal_best_position = torch.where(personal_best_mask, position, personal_best_position)
        personal_best_value = torch.where(personal_best_mask, mse, personal_best_value)

        # Update global best
        global_min_mse, global_min_indices = torch.min(mse, dim=0, keepdim=True)
        global_best_mask = global_min_mse < global_best_value
        global_best_value[global_best_mask] = global_min_mse[global_best_mask]
        global_best_indices_expanded = global_min_indices.expand(-1, 2)
        global_best_updated_position = torch.gather(position, 0, global_best_indices_expanded)
        print(global_best_updated_position.shape)
        global_best_position = torch.where(global_best_mask.unsqueeze(-1), global_best_updated_position,
                                           global_best_position)

        # Update velocity and position
        r1, r2 = torch.rand((num_particles, 2), device=weight.device), torch.rand((num_particles, 2), device=weight.device)
        cognitive_velocity = c1 * r1 * (personal_best_position - position)
        social_velocity = c2 * r2 * (global_best_position - position)
        velocity = w * velocity + cognitive_velocity + social_velocity
        position += velocity

    return global_best_position[:, 0], global_best_position[:, 1]

def pso_asymmetric_quantization(weight, num_bits, num_particles, iterations):
    """
    Particle Swarm Optimization for asymmetric per-channel quantization.

    Args:
    weight (torch.Tensor): The weights of the convolutional layer.
    num_bits (int): The number of bits for quantization.
    num_particles (int): Number of particles in PSO.
    iterations (int): Number of iterations in PSO.

    Returns:
    torch.Tensor: Optimal scale and zero-point for each output channel. Shape: [out_channels, 2]
    """
    qmin = 0
    qmax = 2 ** num_bits - 1
    out_channels = weight.size(0)

    # Reshape weights for per-channel processing
    reshaped_weights = weight.view(out_channels, -1)

    # Initialize particle positions and velocities
    L = torch.max(reshaped_weights, dim=1, keepdim=True)[0] - torch.min(reshaped_weights, dim=1, keepdim=True)[0]
    position = torch.randn((out_channels, num_particles, 2), device=weight.device)
    position[:, :, 0] = torch.max(reshaped_weights, dim=1, keepdim=True)[0] * torch.rand((out_channels, num_particles), device=weight.device)
    position[:, :, 0] = torch.maximum(position[:, :, 0], torch.max(reshaped_weights, dim=1, keepdim=True)[0] - 0.5 * L)
    position[:, :, 1] = torch.min(reshaped_weights, dim=1, keepdim=True)[0]  * torch.rand((out_channels, num_particles), device=weight.device)
    position[:, :, 1] = torch.minimum(position[:, :, 1], torch.min(reshaped_weights, dim=1, keepdim=True)[0] + 0.5 * L)
    velocity = torch.zeros((out_channels, num_particles, 2), device=weight.device)

    # Personal best positions and values
    personal_best_position = position.clone()
    personal_best_value = torch.full((out_channels, num_particles, 1), float('inf'), device=weight.device)

    # Global best positions and values
    global_best_position = torch.zeros((out_channels, 1, 2), device=weight.device)
    global_best_value = torch.full((out_channels, 1, 1), float('inf'), device=weight.device)

    # PSO constants
    c1, c2 = 1.5, 1.5  # cognitive and social constants
    w = 0.5  # inertia weight

    for i in range(iterations):
        # Calculate quantized weights and MSE
        scale = (position[:, :, 0].unsqueeze(-1) - position[:, :, 1].unsqueeze(-1)) / (qmax)
        zero_point = -torch.round(position[:, :, 1].unsqueeze(-1) / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)

        quantized = (torch.clamp(torch.round(reshaped_weights.unsqueeze(1)/ scale)+ zero_point, qmin, qmax) - zero_point)* scale
        mse = torch.mean((reshaped_weights.unsqueeze(1) - quantized) ** 2 * reshaped_weights.unsqueeze(1).abs(), dim=2, keepdim=True)

        # Update personal best
        personal_best_mask = mse < personal_best_value
        personal_best_position = torch.where(personal_best_mask, position, personal_best_position)
        personal_best_value = torch.where(personal_best_mask, mse, personal_best_value)

        # Update global best
        global_min_mse, global_min_indices = torch.min(mse, dim=1, keepdim=True)
        global_best_mask = global_min_mse < global_best_value
        global_best_value[global_best_mask] = global_min_mse[global_best_mask]
        global_best_indices_expanded = global_min_indices.expand(-1, -1, 2)
        global_best_updated_position = torch.gather(position, 1, global_best_indices_expanded).squeeze(1)
        global_best_mask = global_best_mask.squeeze(-1)
        global_best_position = torch.where(global_best_mask.unsqueeze(-1), global_best_updated_position.unsqueeze(1),
                                           global_best_position)

        # Update velocity and position
        r1, r2 = torch.rand((out_channels, num_particles, 2), device=weight.device), torch.rand((out_channels, num_particles, 2), device=weight.device)
        cognitive_velocity = c1 * r1 * (personal_best_position - position)
        social_velocity = c2 * r2 * (global_best_position - position)
        velocity = w * velocity + cognitive_velocity + social_velocity
        position += velocity

    return global_best_position.squeeze(1)[:, 0], global_best_position.squeeze(1)[:, 1]

class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        # if self.sym:
        #     raise NotImplementedError
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        '''mse params'''
        self.scale_method = scale_method
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
                self.delta = torch.nn.Parameter(torch.tensor(delta), requires_grad=True)
                self.zero_point = torch.nn.Parameter(torch.tensor(zero_point), requires_grad=True)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def lp_loss(self, pred, tgt, p=2.0, weight_mask=None):
        if not self.channel_wise:
            if weight_mask is not None:
                x = (pred - tgt).abs().pow(2) * weight_mask
            else:
                x = (pred - tgt).abs().pow(2)
            return x.mean()
        else:
            # tgt = tgt.reshape(tgt.shape[0], -1)
            # pred = pred.reshape(pred.shape[0], -1)
            #
            # temp = torch.quantile(tgt.abs(), q=0.33, dim=-1, keepdim=True)
            # weight = torch.where(tgt.abs() > temp, tgt.abs(), torch.ones_like(tgt) * torch.min(tgt.abs(), -1, keepdim=True)[0])
            x = (pred - tgt).abs().pow(2)  # * tgt.abs()
            y = torch.flatten(x, 1)
            return y.mean(1)

    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def perform_2D_search(self, x, weight_mask=None):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4, weight_mask)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x, weight_mask=None):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4, weight_mask)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        print(self.scale_method)
        if self.scale_method == 'mse':
            if self.one_side_dist is None:
                self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
            if self.one_side_dist != 'no' or self.sym:  # one-side distribution or symmetric value for 1-d search
                best_min, best_max = self.perform_1D_search(x)
            else:  # 2-d search
                best_min, best_max = self.perform_2D_search(x)
        elif self.scale_method == 'wmse':
            if len(x.shape) == 4:
                N, C, H, W = x.shape
                channel_entropy = calculate_channel_entropy(x)
                weight_mask = channel_entropy[None, :, None, None]
            else:
                weight_mask = None
            if self.one_side_dist is None:
                self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
            if self.one_side_dist != 'no' or self.sym:  # one-side distribution or symmetric value for 1-d search
                best_min, best_max = self.perform_1D_search(x, weight_mask)
            else:
                best_min, best_max = self.perform_2D_search(x, weight_mask)
        elif self.scale_method == 'pso':
            if self.channel_wise:
                best_max, best_min = pso_asymmetric_quantization(x, self.n_bits, 100, 100)
            else:
                exit()
        elif self.scale_method == 'max':
            if self.channel_wise:
                y = torch.flatten(x, 1)
                best_min, best_max = torch._aminmax(y, 1)
            else:
                best_min, best_max = torch._aminmax(x)
        else:
            exit()
        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return 'wbit={}, abit={}, disable_act_quant={}'.format(
            self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant
        )
