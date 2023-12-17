from nas.search_algo.arch_manager import ArchManager
from nas.efficiency_predictor.latency import LatencyPredictor
from nas.efficiency_predictor.flops import FLOPsPredictor
import copy
import random
import numpy as np
from tqdm import tqdm


class EvolutionSearch:
    def __init__(
        self,
        flops_predictor: FLOPsPredictor,
        efficiency_predictor: LatencyPredictor,
        arch_manager: ArchManager
    ):
        self.flops_predictor = flops_predictor
        self.efficiency_predictor = efficiency_predictor
        self.arch_manager = arch_manager

        self.mutate_prob = 0.1
        self.population_size = 100
        self.max_time_budget = 500
        self.parent_ratio = 0.25
        self.mutate_ratio = 0.5

    def random_spec_by_constraint(self, constraint):
        while True:
            spec = self.arch_manager.random_spec()
            efficiency = self.efficiency_predictor.predict_efficiency_from_spec(
                spec)
            if efficiency <= constraint:
                return spec, efficiency

    def mutate_spec(self, spec, constraint):
        while True:
            new_spec = copy.deepcopy(spec)
            self.arch_manager.mutate_spec(new_spec, self.mutate_prob)
            efficiency = self.efficiency_predictor.predict_efficiency_from_spec(
                new_spec)

            if efficiency < constraint:
                return new_spec, efficiency

    def crossover_spec(self, spec1, spec2, constraint):
        while True:
            new_spec = copy.deepcopy(spec1)
            for key in new_spec.keys():
                if not isinstance(new_spec[key], list):
                    new_spec[key] = random.choice([spec1[key], spec2[key]])
                else:
                    for i in range(len(new_spec[key])):
                        new_spec[key][i] = random.choice(
                            [spec1[key][i], spec2[key][i]])

            efficiency = self.efficiency_predictor.predict_efficiency_from_spec(
                new_spec)

            if efficiency < constraint:
                return new_spec, efficiency

    def run_search(self, constraint):
        n_mutate = int(round(self.mutate_ratio * self.population_size))
        n_parent = int(round(self.parent_ratio * self.population_size))

        print('Generating random population ...')
        population = []  # element: (flops, spec, latency)
        for _ in range(self.population_size):
            spec, efficiency = self.random_spec_by_constraint(
                constraint)
            flops = self.flops_predictor.predict_flops(spec)
            population.append((flops, spec, efficiency))

        print('Start evolution')
        best_flops = [0]
        best_info = None
        with tqdm(
            desc=f'Constraint {constraint} ms',
            total=self.max_time_budget
        ) as t:
            for _ in range(self.max_time_budget):
                parents = sorted(population, key=lambda x: x[0])[
                    ::-1][:n_parent]
                flops = parents[0][0]
                t.set_postfix({'Flops': '%.4f' % flops})

                if flops > best_flops[-1]:
                    best_flops.append(flops)
                    best_info = parents[0]
                else:
                    best_flops.append(best_flops[-1])

                population = parents
                for _ in range(n_mutate):
                    par_spec = population[np.random.randint(n_parent)][1]
                    mutated_spec, mutated_efficiency = self.mutate_spec(
                        par_spec, constraint)
                    mutated_flops = self.flops_predictor.predict_flops(
                        mutated_spec)
                    population.append(
                        (mutated_flops, mutated_spec, mutated_efficiency))

                for _ in range(self.population_size - n_mutate):
                    par_spec1 = population[np.random.randint(n_parent)][1]
                    par_spec2 = population[np.random.randint(n_parent)][1]

                    cross_spec, cross_efficiency = self.crossover_spec(
                        par_spec1, par_spec2, constraint)
                    cross_flops = self.flops_predictor.predict_flops(
                        cross_spec)
                    population.append(
                        (cross_flops, cross_spec, cross_efficiency))

                t.update(1)

        return best_flops, best_info
