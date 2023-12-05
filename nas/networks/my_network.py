import torch.nn as nn
from nas.utils import utils
from nas.networks.my_module import MyModule
from nas.networks.my_layer import MyConv2D


class MyNetwork(MyModule):
    CHANNEL_DIVISIBLE = 8

    ''' Inherited methods from MyModule'''

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    ''' Implemented methods '''

    def set_bn_param(self, momentum, eps, gn_channel_per_group=None, ws_eps=None):
        self.replace_bn_with_gn(gn_channel_per_group)

        for m in self.modules():
            if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
                m.momentum = momentum
                m.eps = eps
            elif isinstance(m, nn.GroupNorm):
                m.eps = eps

        self.replace_conv2d_with_my_conv2d(ws_eps)

    def replace_bn_with_gn(self, gn_channel_per_group=None):
        if gn_channel_per_group is None:
            return

        for m in self.modules():
            to_replace_dict = {}
            for name, sub_m in m.named_children():
                if isinstance(sub_m, nn.BatchNorm2d):
                    num_groups = sub_m.num_features // utils.min_divisible_value(
                        sub_m.num_features, gn_channel_per_group
                    )
                    gn_m = nn.GroupNorm(
                        num_groups=num_groups,
                        num_channels=sub_m.num_features,
                        eps=sub_m.eps,
                        affine=True,
                    )

                    # load weight
                    gn_m.weight.data.copy_(sub_m.weight.data)
                    gn_m.bias.data.copy_(sub_m.bias.data)
                    # load requires_grad
                    gn_m.weight.requires_grad = sub_m.weight.requires_grad
                    gn_m.bias.requires_grad = sub_m.bias.requires_grad

                    to_replace_dict[name] = gn_m
            m._modules.update(to_replace_dict)

    def replace_conv2d_with_my_conv2d(self, ws_eps=None):
        if ws_eps is None:
            return

        for m in self.modules():
            to_update_dict = {}
            for name, sub_module in m.named_children():
                if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
                    # only replace conv2d layers that are followed by normalization layers (i.e., no bias)
                    to_update_dict[name] = sub_module
            for name, sub_module in to_update_dict.items():
                m._modules[name] = MyConv2D(
                    sub_module.in_channels,
                    sub_module.out_channels,
                    sub_module.kernel_size,
                    sub_module.stride,
                    sub_module.padding,
                    sub_module.dilation,
                    sub_module.groups,
                    sub_module.bias,
                )
                # load weight
                m._modules[name].load_state_dict(sub_module.state_dict())
                # load requires_grad
                m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
                if sub_module.bias is not None:
                    m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
        # set ws_eps
        for m in self.modules():
            if isinstance(m, MyConv2D):
                m.WS_EPS = ws_eps

    def get_bn_param(self):
        ws_eps = None
        for m in self.modules():
            if isinstance(m, MyConv2D):
                ws_eps = m.WS_EPS
                break
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    "momentum": m.momentum,
                    "eps": m.eps,
                    "ws_eps": ws_eps,
                }
            elif isinstance(m, nn.GroupNorm):
                return {
                    "momentum": None,
                    "eps": m.eps,
                    "gn_channel_per_group": m.num_channels // m.num_groups,
                    "ws_eps": ws_eps,
                }
        return None
