import torch.nn as nn
from collections import OrderedDict
from nas.utils import utils
from nas.networks.my_module import MyModule


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        MyConv2D.__name__: MyConv2D,
        ZeroLayer.__name__: ZeroLayer,
        IdentityLayer.__name__: IdentityLayer,
        MyLinearLayer.__name__: MyLinearLayer,
        MBConvLayer.__name__: MBConvLayer,
        ##########################################################
        MyResBlock.__name__: MyResBlock,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class My2DLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        act_func='relu',
        dropout_rate=0,
        ops_order='weight_bn_act'
    ):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        ''' modules '''
        modules = {}

        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None

        # activation
        modules['act'] = utils.build_activation(
            self.act_func, self.ops_list[0] != 'act' and self.use_bn
        )

        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None

        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    ''' Inherited methods from MyModule'''

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class MyConv2D(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_bn=True,
        act_func='relu',
        dropout_rate=0,
        ops_order='weight_bn_act',
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        super(MyConv2D, self).__init__(
            in_channels, out_channels,
            use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = utils.get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict({
            'conv': nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=padding,
                dilation=self.dilation,
                groups=utils.min_divisible_value(
                    self.in_channels, self.groups),
                bias=self.bias
            )
        })

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        conv_str = 'Conv:\n'
        conv_str += f'\tgroup: {self.groups}\n'
        conv_str += f'\tdilation: {self.dilation}\n'
        conv_str += f'\tin: {self.in_channels}\n'
        conv_str += f'\tout: {self.out_channels}\n'
        conv_str += f'\tk: {kernel_size[0]}x{kernel_size[1]}\n'
        conv_str += f'\ts: {self.stride}\n'
        conv_str += f'\tdropout: {self.dropout_rate}\n'
        conv_str += f'\top-order: {self.ops_order}\n'

        if self.use_bn:
            if isinstance(self.bn, nn.GroupNorm):
                conv_str += f'\tnorm: GN{self.bn.num_groups}\n'
            elif isinstance(self.bn, nn.BatchNorm2d):
                conv_str += '\tnorm: BN\n'

        conv_str += '\tact: ' + self.act_func.upper()

        return conv_str

    @property
    def config(self):
        return {
            'name': MyConv2D.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            **super(MyConv2D, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return MyConv2D(**config)


class MyGlobalAvgPool2D(nn.Module):
    def __init__(self, keep_dim=True):
        super(MyGlobalAvgPool2D, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return 'MyGlobalAvgPool2D(keep_dim=%s)' % self.keep_dim


class ZeroLayer(MyModule):
    def __init__(self):
        super(ZeroLayer, self).__init__()

    ''' Inherited methods from MyModule'''

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer()


class MyLinearLayer(MyModule):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order='weight_bn_act',
    ):
        super(MyLinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        ''' modules '''
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = utils.build_activation(
            self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {
            'linear': nn.Linear(self.in_features, self.out_features, self.bias)
        }

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return f'Linear:\n\tin: {self.in_features}\n\tout: {self.out_features}'

    @property
    def config(self):
        return {
            'name': MyLinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return MyLinearLayer(**config)


class MBConvLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func='relu6',
        use_se=False,
        groups=None,
    ):
        super(MBConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.groups = groups

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'conv',
                            nn.Conv2d(
                                self.in_channels, feature_dim, 1, 1, 0, bias=False
                            ),
                        ),
                        ('bn', nn.BatchNorm2d(feature_dim)),
                        ('act', utils.build_activation(
                            self.act_func, inplace=True)),
                    ]
                )
            )

        pad = utils.get_same_padding(self.kernel_size)
        groups = (
            feature_dim
            if self.groups is None
            else utils.min_divisible_value(feature_dim, self.groups)
        )
        depth_conv_modules = [
            (
                'conv',
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=groups,
                    bias=False,
                ),
            ),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', utils.build_activation(self.act_func, inplace=True)),
        ]

        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                    ('bn', nn.BatchNorm2d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            expand_ratio = self.mid_channels // self.in_channels
            feature_dim = self.mid_channels

        groups = (
            feature_dim if self.groups is None
            else utils.min_divisible_value(feature_dim, self.groups)
        )

        layer_str = f'MBConv:\n'
        layer_str += f'\t\tgroup: {groups}\n'
        layer_str += f'\t\tin: {self.in_channels}\n\t\tout: {self.out_channels}\n'
        layer_str += f'\t\tk: {self.kernel_size}x{self.kernel_size}\n'
        layer_str += f'\t\ts: {self.stride}\n'
        layer_str += f'\t\te: {expand_ratio}\n'
        if isinstance(self.point_linear.bn, nn.GroupNorm):
            layer_str += f'\t\tnorm: GN{self.point_linear.bn.num_groups}\n'
        elif isinstance(self.point_linear.bn, nn.BatchNorm2d):
            layer_str += '\t\tnorm: BN\n'
        layer_str += '\t\tact: ' + self.act_func.upper()

        return layer_str

    @property
    def config(self):
        return {
            'name': MBConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'act_func': self.act_func,
            'use_se': self.use_se,
            'groups': self.groups,
        }

    @staticmethod
    def build_from_config(config):
        return MBConvLayer(**config)


class IdentityLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order='weight_bn_act',
    ):
        super(IdentityLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class MyResBlock(MyModule):
    def __init__(self, conv, shortcut):
        super(MyResBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    ''' Inherited methods from MyModule'''

    def forward(self, x):
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.conv(x)
        else:
            res = self.conv(x) + self.shortcut(x)

        return res

    @property
    def module_str(self):
        layer_str = f'ResNetBlock:\n'
        layer_str += f'\t{self.conv.module_str if self.conv is not None else None}\n'
        layer_str += f'\tShortcut: {self.shortcut.module_str if self.shortcut is not None else None}'
        return layer_str

    @property
    def config(self):
        return {
            'name': MyResBlock.__name__,
            'conv': self.conv.config if self.conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv_config = (
            config['conv'] if 'conv' in config else config['mobile_inverted_conv']
        )
        conv = set_layer_from_config(conv_config)
        shortcut = set_layer_from_config(config['shortcut'])
        return MyResBlock(conv, shortcut)

    @property
    def mobile_inverted_conv(self):
        return self.conv
