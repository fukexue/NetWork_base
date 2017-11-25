import math
import numpy as np


class SigmoidActivator(object):
    def forward1(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class RuleActivator(object):
    def forward(self, weighted_input):
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 1 else 0


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1
