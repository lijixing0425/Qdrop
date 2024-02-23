import torch.nn as nn
from .quant_layer import QuantModule, UniformAffineQuantizer
from models.resnet import BasicBlock, Bottleneck
from models.regnet import ResBottleneckBlock
from models.mobilenetv2 import InvertedResidual
from models.mnasnet import _InvertedResidual
from models.shufflenetv2 import shufflenetv2_block, channel_shuffle
from models.mobilenetv1 import DwsConvBlock
import torch


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        self.activation_function = basic_block.relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class _QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, _inv_res: _InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.apply_residual = _inv_res.apply_residual
        self.conv = nn.Sequential(
            QuantModule(_inv_res.layers[0], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[3], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[6], weight_quant_params, act_quant_params, disable_act_quant=True),
        )
        self.conv[0].activation_function = nn.ReLU()
        self.conv[1].activation_function = nn.ReLU()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

class Quant_shufflenetv2_block(BaseQuantBlock):
    def __init__(self, inv_res: shufflenetv2_block, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(Quant_shufflenetv2_block, self).__init__()
        self.stride = inv_res.stride

        if inv_res.stride > 1:
            self.branch1_conv1 = QuantModule(inv_res.branch1[0], weight_quant_params, act_quant_params)
            self.branch1_conv2 = QuantModule(inv_res.branch1[2], weight_quant_params, act_quant_params, disable_act_quant=True)
            self.branch1_conv2.activation_function = inv_res.branch1[4]
        else:
            self.branch1 = nn.Sequential()

        self.branch2_conv1 = QuantModule(inv_res.branch2[0], weight_quant_params, act_quant_params)
        self.branch2_conv2 = QuantModule(inv_res.branch2[3], weight_quant_params, act_quant_params)
        self.branch2_conv3 = QuantModule(inv_res.branch2[5], weight_quant_params, act_quant_params, disable_act_quant=True)
        self.branch2_conv1.activation_function = inv_res.branch2[2]
        self.branch2_conv3.activation_function = inv_res.branch2[7]

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):

        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.branch2_conv1(x2)
            x2 = self.branch2_conv2(x2)
            x2 = self.branch2_conv3(x2)

            out = torch.cat((x1, x2), dim=1)
        else:

            x1 = self.branch1_conv1(x)
            x1 = self.branch1_conv2(x1)

            x2 = self.branch2_conv1(x)
            x2 = self.branch2_conv2(x2)
            x2 = self.branch2_conv3(x2)
            out = torch.cat((x1, x2), dim=1)

        if self.use_act_quant:
            out = self.act_quantizer(out)

        out = channel_shuffle(out, 2)

        return out

class Quant_mv1_block(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: DwsConvBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}, eq=True):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.dw_conv.conv, weight_quant_params, act_quant_params)
        self.conv1.activation_function = nn.ReLU()

        self.conv2 = QuantModule(bottleneck.pw_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv2.activation_function = nn.ReLU()

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_act_quant:
            x = self.act_quantizer(x)
        return x

specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual,
    shufflenetv2_block: Quant_shufflenetv2_block,
    DwsConvBlock: Quant_mv1_block
}
