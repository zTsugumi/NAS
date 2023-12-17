from nas.efficiency_predictor.latency import LatencyPredictor
from nas.efficiency_predictor.flops import FLOPsPredictor
from nas.networks.mbv3 import MobileNetV3
from nas.search_algo.random_search import RandomSearch

from nas.search_algo.arch_manager import ArchManager
from nas.search_algo.evo import EvolutionSearch

if __name__ == '__main__':
    flops_predictor = FLOPsPredictor()
    latency_predictor = LatencyPredictor()
    arch_manager = ArchManager()

    # finder = RandomSearch(
    #     efficiency_predictor=latency_predictor,
    #     arch_manager=arch_manager)
    finder = EvolutionSearch(
        flops_predictor=flops_predictor,
        efficiency_predictor=latency_predictor,
        arch_manager=arch_manager
    )

    best_spec, _ = finder.run_search(30)    # ms



    # net = MobileNetV3.build_from_spec(best_spec, 2)

    # with open('data/found_net.txt', 'w') as f:
    #     f.write(net.module_str)

    # finder.run_compare()
