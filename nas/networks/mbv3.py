from nas.networks.my_network import MyNetwork
from nas.networks.my_layer import MyConv2D, MyResBlock, MyLinearLayer, MyGlobalAvgPool2D, MBConvLayer, IdentityLayer, set_layer_from_config
from nas.utils import utils


class MobileNetV3(MyNetwork):
    def __init__(
        self,
        first_conv: MyConv2D,
        blocks: list[MyResBlock],
        final_expand_layer: MyConv2D,
        feature_mix_layer: MyConv2D,
        classifier: MyLinearLayer
    ):
        super(MobileNetV3, self).__init__()
        self.first_conv = first_conv
        self.blocks = blocks
        self.final_expand_layer = final_expand_layer
        self.global_avg_pool = MyGlobalAvgPool2D(keep_dim=True)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    ''' Inherited methods from MyModule '''

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = self.global_avg_pool(x)
        x = self.feature_mix_layer(x)
        x = x.view(x.size[0], -1)
        x = self.classifier(x)

        return x

    @property
    def module_str(self):
        res = self.first_conv.module_str + '\n'
        for block in self.blocks:
            res += block.module_str + '\n'
        res += self.final_expand_layer.module_str + '\n'
        res += self.global_avg_pool.__repr__() + '\n'
        res += self.feature_mix_layer.module_str + '\n'
        res += self.classifier.module_str

        return res

    @property
    def config(self):
        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [block.config for block in self.blocks],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config: dict):
        first_conv = set_layer_from_config(config['first_conv'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(set_layer_from_config(block_config))

        final_expand_layer = set_layer_from_config(
            config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(
            config['feature_mix_layer'])
        classifier = set_layer_from_config(
            config['classifier'])

        net = MobileNetV3(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )

        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    @staticmethod
    def build_from_spec(spec: dict, n_classes):
        base_stage_width = [24, 40, 80, 112, 160, 960, 1280]
        stride_stages = [2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'relu', 'relu']

        width_mult = 1.0
        width_list = []
        for base_width in base_stage_width[:-2]:
            width_list.append(utils.make_divisible(base_width*width_mult,
                                                   MyNetwork.CHANNEL_DIVISIBLE))

        build_config = {}
        # First conv
        feature_dim = utils.make_divisible(16, MyNetwork.CHANNEL_DIVISIBLE)
        build_config['first_conv'] = {
            'name': MyConv2D.__name__,
            'in_channels': 3,
            'out_channels': feature_dim,
            'kernel_size': 3,
            'stride': 2,
            'act_func': 'relu'
        }

        # First inverted block
        build_config['blocks'] = [{
            'name': MyResBlock.__name__,
            'mobile_inverted_conv': {
                'name': MBConvLayer.__name__,
                'in_channels': feature_dim,
                'out_channels': feature_dim,
                'kernel_size': 3,
                'expand_ratio': 1,
                'stride': 1,
                'act_func': 'relu',
            },
            'shortcut': {
                'name': IdentityLayer.__name__,
                'in_channels': feature_dim,
                'out_channels': feature_dim,
            }
        }]

        # Next inverted blocks
        for i in range(20):
            stage = i // 4
            depth_max = spec['d'][stage]
            depth_cur = i % 4 + 1
            if depth_cur > depth_max:
                continue

            stride = stride_stages[stage] if depth_cur == 1 else 1
            build_config['blocks'].append({
                'name': MyResBlock.__name__,
                'mobile_inverted_conv': {
                    'name': MBConvLayer.__name__,
                    'in_channels': feature_dim,
                    'out_channels': width_list[stage],
                    'kernel_size': spec['ks'][i],
                    'expand_ratio': spec['e'][i],
                    'stride': stride,
                    'act_func': act_stages[stage],
                },
                'shortcut': {
                    'name': IdentityLayer.__name__,
                    'in_channels': feature_dim,
                    'out_channels': feature_dim,
                }
                if feature_dim == width_list[stage] and stride == 1
                else None
            })

            feature_dim = width_list[stage]

        # final expand layer, feature mix layer & classifier
        final_expand_width = utils.make_divisible(
            base_stage_width[-2] * width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        build_config['final_expand_layer'] = {
            'name': MyConv2D.__name__,
            'in_channels': feature_dim,
            'out_channels': final_expand_width,
            'kernel_size': 1,
            'act_func': 'relu'
        }

        last_channel = utils.make_divisible(
            base_stage_width[-1] * width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        build_config['feature_mix_layer'] = {
            'name': MyConv2D.__name__,
            'in_channels': final_expand_width,
            'out_channels': last_channel,
            'kernel_size': 1,
            'bias': False,
            'use_bn': False,
            'act_func': 'relu'
        }

        build_config['classifier'] = {
            'name': MyLinearLayer.__name__,
            'in_features': last_channel,
            'out_features': n_classes,
        }

        net = MobileNetV3.build_from_config(build_config)

        return net


# class DynamicMobileNetV3(MobileNetV3):
#     def __init__(
#         self,
#         n_classes=1000,
#         bn_param=(0.1, 1e-5),
#         dropout_rate=0.1,
#         width_mult=1.0,
#         ks_list=[3],
#         expand_ratio_list=[6],
#         depth_list=[4],
#     ):
#         self.width_mult = width_mult
#         self.ks_list = ks_list.sort()
#         self.expand_ratio_list = expand_ratio_list.sort()
#         self.depth_list = depth_list.sort()

#         base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

#         final_expand_width = utils.make_divisible(
#             base_stage_width[-2] * self.width_mult,
#             MyNetwork.CHANNEL_DIVISIBLE
#         )

#         last_channel = utils.make_divisible(
#             base_stage_width[-1] * self.width_mult,
#             MyNetwork.CHANNEL_DIVISIBLE
#         )

#         stride_stages = [1, 2, 2, 2, 1, 2]
#         act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
#         n_block_list = [1] + [max(self.depth_list)] * 5
#         width_list = []
#         for base_width in base_stage_width[:-2]:
#             width = utils.make_divisible(
#                 base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
#             )
#             width_list.append(width)

#         input_channel, first_block_dim = width_list[0], width_list[1]
#         first_conv = MyConv2D(
#             3, input_channel, kernel_size=3, stride=2, act_func='h_swish'
#         )

#         first_block = MyResBlock(
#             conv=MBConvLayer(
#                 in_channels=input_channel,
#                 out_channels=first_block_dim,
#                 kernel_size=3,
#                 stride=stride_stages[0],
#                 expand_ratio=1,
#                 act_func=act_stages[0],
#             ),
#             shortcut=IdentityLayer(first_block_dim, first_block_dim)
#             if input_channel == first_block_dim
#             else None
#         )

#         self.block_group_info = []
#         blocks = [first_block]
#         _block_index = 1
#         feature_dim = first_block_dim

#         for width, n_block, s, act_func in zip(
#             width_list[2:],
#             n_block_list[1:],
#             stride_stages[1:],
#             act_stages[1:],
#         ):
#             self.block_group_info.append(
#                 [_block_index + i for i in range(n_block)]
#             )

#             _block_index += n_block

#             output_channel = width
#             for i in range(n_block):
#                 if i == 0:
#                     stride = s
#                 else:
#                     stride = 1

#                 mobile_inverted_conv = DynamicMBConvLayer(
#                     in_channel_list=val2list(feature_dim),
#                     out_channel_list=val2list(output_channel),
#                     kernel_size_list=ks_list,
#                     expand_ratio_list=expand_ratio_list,
#                     stride=stride,
#                     act_func=act_func,
#                     use_se=use_se,
#                 )

#         super(DynamicMobileNetV3, self).__init__(
#             first_conv,
#             blocks,
#         )
