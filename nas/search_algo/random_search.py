from nas.search_algo.arch_manager import ArchManager
from nas.efficiency_predictor.latency import LatencyPredictor
from nas.networks.mbv3 import MobileNetV3


class RandomSearch:
    def __init__(
        self,
        efficiency_constraint: float,
        efficiency_predictor: LatencyPredictor,
    ):
        self.efficiency_constraint = efficiency_constraint
        self.efficiency_predictor = efficiency_predictor
        self.arch_manager = ArchManager()

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint

    def run_search(self):
        while True:
            spec = self.arch_manager.random_spec()
            efficiency = self.efficiency_predictor.predict_efficiency_from_spec(
                spec)
            if efficiency <= self.efficiency_constraint:
                return spec, efficiency

    def run_compare(self, plot=False):
        efficiency = {
            'Est': [],
            'Truth': [],
        }
        x = []
        diff = []

        with open('data/comparison.txt', 'w') as f:
            for i in range(1000):
                x.append(i)
                spec = self.arch_manager.random_spec()
                net = MobileNetV3.build_from_spec(spec, 1000)

                efficiency['Est'].append(
                    round(self.efficiency_predictor.predict_efficiency_from_spec(spec), 2) / 100)
                efficiency['Truth'].append(
                    round(self.efficiency_predictor.predict_efficiency_from_net(
                        net, spec.get('r', [224])[0]), 2))
                diff.append(
                    abs(efficiency['Est'][-1] - efficiency['Truth'][-1]))

                f.write(
                    f'Try {i} - Est: {efficiency["Est"][-1]:.2f}\tTruth: {efficiency["Truth"][-1]:.2f}\n')

            import numpy as np
            mae = np.mean(diff)
            f.write(f'MAE = {mae}')

        if plot:
            import matplotlib.pyplot as plt
            import numpy as np

            x = np.arange(len(x))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0

            fig, ax = plt.subplots(layout='constrained')
            for attribute, measurement in efficiency.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                ax.bar_label(rects, padding=3)
                multiplier += 1

            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency Comparison')
            ax.set_xticks(x + width, x)
            ax.legend(loc='upper left', ncols=3)

            plt.show()
            fig.savefig('data/comparison.png')
