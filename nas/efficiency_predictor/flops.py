import torch
import copy
import numpy as np
from nas.networks.my_layer import MyConv2D, MyResBlock, MyLinearLayer, MBConvLayer, IdentityLayer
from nas.utils import utils


class FLOPsPredictor:
    def __init__(self,
                 device='cuda:0',
                 multiplier=1.2,
                 batch_size=64,
                 load_table=None):
        self.multiplier = multiplier
        self.batch_size = batch_size
        self.device = device
        self.efficiency_dict = {}

        if load_table is not None:
            self.efficiency_dict = np.load(
                load_table, allow_pickle=True
            ).item()
        else:
            self.build_lut(batch_size)

    def build_lut(self, batch_size=1, resolutions=[160, 176, 192, 208, 224]):
        for resolution in resolutions:
            self.build_single_lut(batch_size, resolution)

        np.save('data/flops_lut.npy', self.efficiency_dict)

    def build_single_lut(self, batch_size=1, resolution=224):
        # block, input_size, in_channels, out_channels, expand_ratio, kernel_size, stride, act
        configs = [
            (MyConv2D, resolution, 3, 16, 3, 2, 'relu'),
            (MyResBlock, resolution//2, 16, 16,
             [1], [3, 5, 7], 1, 'relu'),
            (MyResBlock, resolution//2, 16, 24,
             [3, 4, 6], [3, 5, 7], 2, 'relu'),
            (MyResBlock, resolution//4, 24, 24,
             [3, 4, 6], [3, 5, 7], 1, 'relu'),
            (MyResBlock, resolution//4, 24, 24,
             [3, 4, 6], [3, 5, 7], 1, 'relu'),
            (MyResBlock, resolution//4, 24, 24,
             [3, 4, 6], [3, 5, 7], 1, 'relu'),
            (MyResBlock, resolution//4, 24, 40,
             [3, 4, 6], [3, 5, 7], 2, 'relu'),
            (MyResBlock, resolution//8, 40, 40,
             [3, 4, 6], [3, 5, 7], 1, 'relu'),
            (MyResBlock, resolution//8, 40, 40,
             [3, 4, 6], [3, 5, 7], 1, 'relu'),
            (MyResBlock, resolution//8, 40, 40,
             [3, 4, 6], [3, 5, 7], 1, 'relu'),
            (MyResBlock, resolution//8, 40, 80,
             [3, 4, 6], [3, 5, 7], 2, 'h_swish'),
            (MyResBlock, resolution//16, 80, 80,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//16, 80, 80,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//16, 80, 80,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//16, 80, 112,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//16, 112, 112,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//16, 112, 112,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//16, 112, 112,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//16, 112, 160,
             [3, 4, 6], [3, 5, 7], 2, 'h_swish'),
            (MyResBlock, resolution//32, 160, 160,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//32, 160, 160,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyResBlock, resolution//32, 160, 160,
             [3, 4, 6], [3, 5, 7], 1, 'h_swish'),
            (MyConv2D, resolution // 32, 160, 960, 1, 1, 'h_swish'),
            (MyConv2D, 1, 960, 1280, 1, 1, 'h_swish'),
            (MyLinearLayer, 1, 1280, 1000, 1, 1),
        ]

        efficiency_dict = {
            'mobile_inverted_blocks': [],
            'other_blocks': {}
        }

        for layer_idx in range(len(configs)):
            config = configs[layer_idx]
            op_type = config[0]
            if op_type == MyResBlock:
                _, input_size, in_channels, out_channels, expand_list, ks_list, stride, act = config
                in_channels = int(round(in_channels * self.multiplier))
                out_channels = int(round(out_channels * self.multiplier))
                template_config = {
                    'name': MyResBlock.__name__,
                    'mobile_inverted_conv': {
                        'name': MBConvLayer.__name__,
                        'in_channels': in_channels,
                        'out_channels': out_channels,
                        'kernel_size': None,
                        'expand_ratio': None,
                        'stride': stride,
                        'act_func': act,
                    },
                    'shortcut': {
                        'name': IdentityLayer.__name__,
                        'in_channels': in_channels,
                        'out_channels': out_channels,
                    }
                    if (in_channels == out_channels and stride == 1)
                    else None,
                }
                sub_dict = {}
                for ks in ks_list:
                    for e in expand_list:
                        build_config = copy.deepcopy(template_config)
                        build_config["mobile_inverted_conv"]["expand_ratio"] = e
                        build_config["mobile_inverted_conv"]["kernel_size"] = ks
                        layer = MyResBlock.build_from_config(build_config)
                        input_shape = (batch_size, in_channels,
                                       input_size, input_size)

                        flop_count = self.measure_single_layer_flops(
                            layer, input_shape) / batch_size

                        sub_dict[(ks, e)] = flop_count
                efficiency_dict["mobile_inverted_blocks"].append(sub_dict)

            elif op_type == MyConv2D:
                _, input_size, in_channels, out_channels, kernel_size, stride, activation = config
                in_channels = int(round(in_channels * self.multiplier))
                out_channels = int(round(out_channels * self.multiplier))
                build_config = {
                    # 'name': ConvLayer.__name__,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "dilation": 1,
                    "groups": 1,
                    "bias": False,
                    "use_bn": True,
                    "act_func": activation,
                }
                layer = MyConv2D.build_from_config(build_config)
                input_shape = (batch_size, in_channels, input_size, input_size)
                flop_count = self.measure_single_layer_flops(
                    layer, input_shape) / batch_size
                efficiency_dict["other_blocks"][layer_idx] = flop_count

            elif op_type == MyLinearLayer:
                _, input_size, in_channels, out_channels, kernel_size, stride = config
                in_channels = int(round(in_channels * self.multiplier))
                out_channels = int(round(out_channels * self.multiplier))
                build_config = {
                    # 'name': LinearLayer.__name__,
                    "in_features": in_channels,
                    "out_features": out_channels,
                }
                layer = MyLinearLayer.build_from_config(build_config)
                input_shape = (batch_size, in_channels)
                flop_count = self.measure_single_layer_flops(
                    layer, input_shape) / batch_size
                efficiency_dict["other_blocks"][layer_idx] = flop_count

            else:
                raise NotImplementedError

        self.efficiency_dict[resolution] = efficiency_dict

    def measure_single_layer_flops(self, layer, input_shape):
        import thop

        inputs = torch.randn(*input_shape, device=self.device)
        network = layer.to(self.device)
        layer.eval()
        utils.rm_bn_from_net(layer)
        flops, _ = thop.profile(network, (inputs,), verbose=False)
        return flops / 1e6

    def predict_efficiency(self, spec: dict):
        input_size = spec.get('r', [224])[0]
        total_stats = 0

        for i in range(20):
            stage = i//4
            depth_max = spec['d'][stage]
            depth_cur = i % 4 + 1
            if depth_cur > depth_max:
                continue
            ks, e = spec['ks'][i], spec['e'][i]

            total_stats += self.efficiency_dict[input_size]['mobile_inverted_blocks'][i+1][(
                ks, e)]

        for key in self.efficiency_dict[input_size]["other_blocks"]:
            total_stats += self.efficiency_dict[input_size]["other_blocks"][key]

        total_stats += self.efficiency_dict[input_size]["mobile_inverted_blocks"][0][
            (3, 1)
        ]

        return total_stats
