from nas.efficiency_predictor.flops import FLOPsPredictor
from nas.efficiency_predictor.latency import LatencyPredictor
from nas.search_algo.random_search import RandomSearch
from nas.networks.mbv3 import MobileNetV3

if __name__ == '__main__':
    latency_predictor = LatencyPredictor()

    finder = RandomSearch(
        efficiency_constraint=25,  # ms
        efficiency_predictor=latency_predictor)
    best_config, _ = finder.run_search()

    net = MobileNetV3.build_from_spec(best_config, 1000)
