from typing import Union
from nas.search_algo.arch_manager import ArchManager
from nas.efficiency_predictor.flops import FLOPsPredictor
from nas.efficiency_predictor.latency import LatencyPredictor
from nas.networks.mbv3 import MobileNetV3


class RandomSearch:
    def __init__(
        self,
        efficiency_constraint: float,
        efficiency_predictor: Union[FLOPsPredictor, LatencyPredictor],
    ):
        self.efficiency_constraint = efficiency_constraint
        self.efficiency_predictor = efficiency_predictor
        self.arch_manager = ArchManager()

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint

    def run_search(self):
        while True:
            spec = self.arch_manager.random_spec()
            efficiency = self.efficiency_predictor.predict_efficiency(spec)
            if efficiency <= self.efficiency_constraint:
                return spec, efficiency

    def run_compare(self):
        for _ in range(10):
            spec = self.arch_manager.random_spec()
            net = MobileNetV3.build_from_spec(spec, 1000)

            efficiency_spec = self.efficiency_predictor.predict_efficiency_from_spec(
                spec)
            efficiency_net = self.efficiency_predictor.predict_efficiency_from_net(
                net, spec.get('r', [224])[0]
            )
            diff = efficiency_spec / efficiency_net * 100

            print(
                f'Est: {efficiency_spec:.4f}\tTruth: {efficiency_net:.4f}\tDiff: {diff:.2f}%')
