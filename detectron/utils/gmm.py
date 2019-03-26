from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from functools import reduce

from detectron.core.config import cfg

from scipy.stats import norm
from scipy.optimize import basinhopping

class GMM:
    def __init__(self, mus, sigmas, phis):
        self.norms = []
        self.mus = mus
        self.phis = phis
        for mu, sigma in zip(mus, sigmas):
            self.norms.append(norm(loc=mu, scale=sigma))

    def predict(self, x):
        p = 0.
        for n, phi in zip(self.norms, self.phis):
            p += n.pdf(x) * phi
        return p

    def max(self):
        # use method L-BFGS-B because the problem is smooth and bounded
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=[(self.mus.min(), self.mus.max())])
        f = lambda x: -self.predict(x)
        res = basinhopping(f, self.mus.mean(), minimizer_kwargs=minimizer_kwargs)
        return res.x

    def exhaustive(self):
        pts = np.linspace(self.mus.min(), self.mus.max(), 800)
        return pts[self.predict(pts).argmax()]

