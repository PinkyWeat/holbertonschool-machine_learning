#!/usr/bin/env python3
""" Hyperparameter Tuning """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ performs Bayesian optimization on noiseless 1D Gaussian process """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ calculates the next best sample location """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best_y = np.min(self.gp.Y)
            improv = best_y - mu - self.xsi
        else:
            best_y = np.max(self.gp.Y)
            improv = mu - best_y - self.xsi

        Z = improv / sigma
        ei = improv * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """ optimizes the black-box function """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            self.gp.update(X_next, self.f(X_next))

        if self.minimize:
            opt = np.argmin(self.gp.Y)
        else:
            opt = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1, :]
        X_opt = self.gp.X[opt]
        Y_opt = self.gp.Y[opt]

        return X_opt, Y_opt
