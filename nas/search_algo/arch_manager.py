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

        sample = {
            'ks': ks,
            'e': e,
            'd': d,
            'r': [random.choice(self.resolutions)]
        }

        return sample
