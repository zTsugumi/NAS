from nas.utils import utils
from nas.networks.my_network import MyNetwork
from nas.networks.my_layer import MyConv2D, MyResBlock, MyLinearLayer
from nas.efficiency_predictor.predictor import Predictor


class LatencyPredictor(Predictor):
    def measure_latency_by_op(self, op_type, config, with_relu=True):
        latency = 0

        if op_type == MyConv2D:
            input_size, in_channels, out_channels, kernel_size, stride, _ = config
            # conv
            x1 = in_channels
            x2 = (kernel_size / stride)**2 * out_channels
            latency += 6.03e-5 * x1 + 1.24e-4 * x2 + 1.89e-1

            # bn + act(default to relu)
            x = input_size
            latency += 6.59e-5 * x + 7.80e-2
            if with_relu:
                latency += 5.6e-6 * x + 5.69e-2

            # pad - WIP

        elif op_type == MyLinearLayer:
            in_channels, out_channels = config
            x1 = in_channels
            x2 = out_channels

            latency += 1.07e-4 * x1 + 1.83e-4 * x2 + 0.164

        else:
            raise NotImplementedError

        return latency

    def measure_single_layer_latency(self, config):
        op_type = config[0]
        latency = 0

        if op_type == MyConv2D:
            latency += self.measure_latency_by_op(
                op_type, config[1:]
            )

        elif op_type == MyResBlock:
            input_size, in_channels, out_channels, expand_ratio, kernel_size, stride, _ = config[
                1:]

            # MBConvLayer
            mid_channels = round(in_channels * expand_ratio)

            if expand_ratio > 1:  # inverted bottleneck
                latency += self.measure_latency_by_op(
                    MyConv2D, (input_size, in_channels, mid_channels,
                               1, 1, None),
                )

            # depth-wise conv - WIP - use conv formula for now
            latency += self.measure_latency_by_op(
                MyConv2D, (input_size, mid_channels, mid_channels,
                           kernel_size, stride, None),
            )

            # point-linear
            latency += self.measure_latency_by_op(
                MyConv2D, (input_size // stride, mid_channels, out_channels,
                           1, 1, None), False
            )

            # IdentityLayer (Shortcut) - WIP

        elif op_type == MyLinearLayer:
            latency += self.measure_latency_by_op(
                op_type, config[1:], False
            )

        return latency

    def get_config_from_spec(self, spec: dict):
        input_size = spec.get('r', [224])[0]
        base_stage_width = [24, 40, 80, 112, 160, 960, 1280]
        stride_stages = [2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'relu', 'relu']

        width_mult = 1.0
        width_list = []
        for base_width in base_stage_width[:-2]:
            width_list.append(utils.make_divisible(base_width*width_mult,
                                                   MyNetwork.CHANNEL_DIVISIBLE))

        # block, input_size, in_channels, out_channels, expand_ratio, kernel_size, stride, act
        feature_dim = utils.make_divisible(16, MyNetwork.CHANNEL_DIVISIBLE)
        configs = [
            (MyConv2D, input_size, 3, feature_dim,                          # first conv
             3, 2, 'relu'),
            (MyResBlock, input_size // 2, feature_dim, feature_dim,         # first inverted block
             1, 3, 1, 'relu')
        ]

        input_size //= 2

        for i in range(20):                                 # next inverted blocks
            stage = i // 4
            depth_max = spec['d'][stage]
            depth_cur = i % 4 + 1
            if depth_cur > depth_max:
                continue

            stride = stride_stages[stage] if depth_cur == 1 else 1
            configs.append((
                MyResBlock, input_size, feature_dim, width_list[stage],
                spec['e'][i], spec['ks'][i], stride, act_stages[stage]
            ))

            feature_dim = width_list[stage]
            input_size //= stride

        # final expand layer, feature mix layer & classifier
        final_expand_width = utils.make_divisible(
            base_stage_width[-2] * width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        configs.append((
            MyConv2D, input_size, feature_dim, final_expand_width, 1, 1, 'relu'
        ))

        last_channel = utils.make_divisible(
            base_stage_width[-1] * width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        configs.append((
            MyConv2D, input_size, final_expand_width, last_channel, 1, 1, 'relu'
        ))

        configs.append((
            MyLinearLayer, last_channel, 1000
        ))

        return configs

    def predict_efficiency(self, spec: dict):
        configs = self.get_config_from_spec(spec)
        total_stats = 0
        for config in configs:
            total_stats += self.measure_single_layer_latency(config)

        return total_stats
