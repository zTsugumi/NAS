from typing import Union
from nas.search_algo.arch_manager import ArchManager
from nas.efficiency_predictor.flops import FLOPsPredictor
from nas.efficiency_predictor.latency import LatencyPredictor


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
