#NHECBE

import math

class LaplaceDistribution:

    def __init__(self, rand, loc, scale):
        self.scale = scale
        self.loc = loc
        self.rand = rand


    def pdf(self, x):
        return (1 / (2 * self.scale)) * math.exp(-abs(x - self.loc) / self.scale)

    def cdf(self, x):
        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if 0 <= p <= 1:
            if p < 0.5:
                return self.loc + self.scale * math.log(2 * p)
            else:
                return self.loc - self.scale * math.log(2 * (1 - p))
        else:
            raise ValueError("p must be between 0 and 1")

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return self.loc

    def variance(self):
        return 2 * self.scale ** 2

    def skewness(self):
        return 0.0

    def ex_kurtosis(self):
        return 3.0

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x >= self.scale:
            return self.shape * (self.scale ** self.shape) / (x ** (self.shape + 1))
        else:
            return 0

    def cdf(self, x):
        if x >= self.scale:
            return 1 - (self.scale / x) ** self.shape
        else:
            return 0

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.scale / ((1 - p) ** (1 / self.shape))
        else:
            raise ValueError("p must be between 0 and 1")

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        if self.shape > 1:
            return self.shape * self.scale / (self.shape - 1)
        else:
            return Exception("Moment undefined")

    def variance(self):
        if self.shape > 2:
            return self.scale ** 2 * self.shape / ((self.shape - 1) ** 2 * (self.shape - 2))
        else:
            return Exception("Moment undefined")

    def skewness(self):
        if self.shape > 3:
            return 2 * (1 + self.shape) / (self.shape - 3) * math.sqrt((self.shape - 2) / self.shape)
        else:
            raise Exception("Moment undefined")

    def ex_kurtosis(self):
        if self.shape > 4:
            return 6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2) / (self.shape * (self.shape - 3) * (self.shape - 4))
        else:
            raise Exception("Moment undefined")

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]
