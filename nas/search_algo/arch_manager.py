import random


class ArchManager:
    def __init__(self):
        self.n_blocks = 20
        self.n_stages = 5
        self.kernel_sizes = [3, 5, 7]
        self.expand_ratios = [3, 4, 6]
        self.depths = [2, 3, 4]
        self.resolutions = [160, 176, 192, 208, 224]

    def random_spec(self):
        ks = []
        d = []
        e = []
        for _ in range(self.n_stages):
            d.append(random.choice(self.depths))

        for _ in range(self.n_blocks):
            e.append(random.choice(self.expand_ratios))
            ks.append(random.choice(self.kernel_sizes))

        spec = {
            'ks': ks,
            'e': e,
            'd': d,
            'r': [random.choice(self.resolutions)]
        }

        return spec

    def mutate_spec(self, spec, mutate_prob):
        for i in range(self.n_blocks):
            if random.random() < mutate_prob:
                spec['ks'][i] = random.choice(self.kernel_sizes)
                spec['e'][i] = random.choice(self.expand_ratios)

        for i in range(self.n_stages):
            if random.random() < mutate_prob:
                spec['d'][i] = random.choice(self.depths)
