#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import parametros as par


def mapa_E_vs_r(bias, I0, X0, N):
    u'''retorna I, X com E(r)  '''
    alpha = par.alpha
    E = par.fit_E_vs_r(bias, ri=0.6, rf=1.6, grau=2)
    I = np.ones(N, float) * I0
    X = np.ones(N, float) * X0
    n = 0
    while n < N - 1:
        I[n + 1] = I[n] + alpha * np.sin(2 * np.pi * X[n])
        if (I[n + 1] + par.b**2) <= .0:
            break
        else:
            r = np.sqrt(((par.R_ext**2 - par.R_int**2) * I[n + 1]) + par.R_int**2)
            beta = par.alpha_1 * (par.beta_1 - E(r))
            X[n + 1] = np.mod(X[n] + beta / np.sqrt(I[n + 1] + par.b**2), 1)
            n += 1
    return I[:n], X[:n]


def mapa_E_vs_r_limitado(bias, I0, X0, N):
    u'''retorna I, X com E(r)  '''
    alpha = par.alpha
    E = par.fit_E_vs_r(bias, ri=0.6, rf=1.6, grau=2)
    I = np.ones(N, float) * I0
    X = np.ones(N, float) * X0
    n = 0
    while n < N - 1:
        I[n + 1] = I[n] + alpha * np.sin(2 * np.pi * X[n])
        if (I[n + 1] + par.b**2) <= .0 or I[n + 1] > 1:
            break
        else:
            r = np.sqrt(((par.R_ext**2 - par.R_int**2) * I[n + 1]) + par.R_int**2)
            beta = par.alpha_1 * (par.beta_1 - E(r))
            X[n + 1] = np.mod(X[n] + beta / np.sqrt(I[n + 1] + par.b**2), 1)
            n += 1
    return I[:n], X[:n]