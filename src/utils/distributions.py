import math
import pyerf
import scipy.special


class UniformDistribution:
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.a:
            return 0.0
        elif self.a <= x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1.0

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.a + p * (self.b - self.a)
        else:
            raise ValueError("p must be between 0 and 1")

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return (self.a + self.b) / 2

    def median(self):
        return (self.a + self.b) / 2

    def variance(self):
        if self.a == self.b:
            raise Exception("Moment undefined")

        return (self.b - self.a) ** 2 / 12

    def skewness(self):
        if self.a == self.b:
            raise Exception("Moment undefined")

        return 0.0

    def ex_kurtosis(self):
        if self.a == self.b:
            raise Exception("Moment undefined")

        ex_kurtosis = 9 / 5 - 3
        return ex_kurtosis

    def mvsk(self):
        if self.a == self.b:
            raise Exception("Moments undefined")

        moments = [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

        return moments


class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        exp_term = -0.5 * ((x - self.loc) / math.sqrt(self.scale)) ** 2
        return (1 / math.sqrt((self.scale * 2 * math.pi))) * math.exp(exp_term)

    def cdf(self, x):
        z = (x - self.loc) / math.sqrt(self.scale)
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def ppf(self, p):
        if 0 <= p <= 1:
            z = math.sqrt(2) * pyerf.erfinv(2 * p - 1)
            return self.loc + z * math.sqrt(self.scale)
        else:
            raise ValueError("p must be between 0 and 1")

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return self.loc

    def median(self):
        return self.loc

    def variance(self):
        return self.scale

    def skewness(self):
        return 0.0

    def ex_kurtosis(self):
        return 0.0

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / math.pi ) * (self.scale / ((x - self.loc) ** 2 + self.scale ** 2))

    def cdf(self, x):
        return 0.5 + (1 / math.pi) * math.atan((x - self.loc) / self.scale)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.loc + self.scale * math.tan(math.pi * (p - 0.5))
        else:
            raise ValueError("p must be between 0 and 1")

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        raise Exception("Moment undefined")

    def median(self):
        return self.loc

    def variance(self):
        raise Exception("Moment undefined")

    def skewness(self):
        raise Exception("Moment undefined")

    def ex_kurtosis(self):
        raise Exception("Moment undefined")

    def mvsk(self):
        raise Exception("Moments undefined")

class LogisticDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        return math.exp(-(x - self.location) / self.scale) / (self.scale * (1 + math.exp(-(x - self.location) / self.scale)) ** 2)

    def cdf(self, x):
        return 1 / (1 + math.exp(-(x - self.location) / self.scale))

    def ppf(self, p):
        if 0 < p < 1:
            return self.location + self.scale * math.log(p / (1 - p))
        else:
            raise ValueError("p must be between 0 and 1")

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return self.location

    def variance(self):
        return (math.pi ** 2) * (self.scale ** 2) / 3

    def skewness(self):
        return 0.0

    def ex_kurtosis(self):
        return 1.2

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        if x > 0:
            return ((x / 2) ** (self.dof / 2 - 1) * math.exp(-x / 2)) / (2 * scipy.special.gamma(self.dof / 2))
        else:
            return 0.0

    def cdf(self, x):
        return scipy.special.gammainc(self.dof / 2, x / 2)

    def ppf(self, p):
        if 0 < p < 1:
            return 2 * scipy.special.gammaincinv(self.dof / 2, p)
        else:
            raise ValueError("p must be between 0 and 1")

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return self.dof

    def variance(self):
        return 2 * self.dof

    def skewness(self):
        return math.sqrt(8 / self.dof)

    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]
