from pylab import *
from numpy import *

class Saturator:
    def __init__(self):
        self.threshold = 1.0
        self.ratio = 1.0
        self.symmetry = 1.0

        self.update_params()

        return

    def update_params(self):
        self.slope = 1. / self.ratio
        self.symslope = 1.0 + self.symmetry * (self.slope - 1.0)
        return

    def process(self, arr):

        res = []

        # -- saturation
        for e in arr:

            if e > self.threshold:
                o = (e - self.threshold) * self.slope + self.threshold
            elif e < -self.threshold:
                o = -((-e - self.threshold) * self.symslope + self.threshold)
            else:
                o = e

            res.append(o)

        return array(res)

# -- MAIN BEHAVIOUR
def main():

    f0 = 200.0
    fs = 16000.0
    amp = 1.0
    nsamples = int(1.0 * fs)
    sig = amp * sin(2.0 * 3.1415 * f0 / fs * arange(0, nsamples))

    sat = Saturator()
    res = sat.process(sig)

    figure(); plot(sig, lw=1); plot(res, lw=1); show()
    return

if __name__ == "__main__":
    main()