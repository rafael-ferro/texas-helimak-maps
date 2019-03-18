#!/usr/bin/python
# -*- coding: utf-8 -*-

''' CALCULA SIGMA2 - DESLOCAMENTO RADIAL MEDIO '''

import timeit
start_time = timeit.default_timer()

import multiprocessing as mp
import numpy as np

import os
import sys

from progressbar import ProgressBar, Percentage, Bar, ETA

import parametros as par
import plotar

if len(sys.argv) > 1:
    biases = [int(sys.argv[1])]
else:
    biases = np.loadtxt('velocidades.txt', usecols=[0], skiprows=1)
    biases = [int(bias) for bias in biases]

    biases = [7.8,7.9,8.4,8.5]

N_X0 = 10
N_I0 = 100
N_part_ini = N_X0 * N_I0
X1 = 0.4
X2 = 0.6
I1 = 0.9
I2 = 1.0
X0 = np.linspace(X1, X2, N_X0)
I0 = np.linspace(I1, I2, N_I0)
N = int(1e6)

print '\n------------------------------------------------------------'
print 'Sigma^2'


def sigma(bias):
    E = par.fit_E_vs_r(bias)
    sigma = np.zeros(N)
    N_part_fin = np.zeros(N)
    r = np.zeros(2)
    pbar = ProgressBar(widgets=['bias = %s: ' % bias, Percentage(), ' ',
                                Bar(marker='=', left='[', right=']'), ' ', ETA()], term_width=60)
    for x0 in pbar(X0):
        for i0 in I0:
            I = np.ones(N, float) * i0
            X = np.ones(N, float) * x0
            r[0] = np.sqrt(((par.R_ext**2 - par.R_int**2) * I[0]) + par.R_int**2)
            n = 0
            while n < N - 1:
                I[n + 1] = I[n] + par.alpha * np.sin(2 * np.pi * X[n])
                if (I[n + 1] + par.b**2) <= .0:# or I[n + 1] > 1:
                    break
                raio = np.sqrt(((par.R_ext**2 - par.R_int**2) * I[n + 1]) + par.R_int**2)
                beta = par.alpha_1 * (par.beta_1 - E(raio))
                X[n + 1] = np.mod(X[n] + beta / np.sqrt(I[n + 1] + par.b**2), 1)
    
                r[1] = np.sqrt(((par.R_ext**2 - par.R_int**2) * I[n]) + par.R_int**2)
                sigma[n] += (r[1] - r[0])**2
                N_part_fin[n] += 1
                n += 1
    for n in range(N):
        if N_part_fin[n] != 0:
            sigma[n] = sigma[n] / N_part_fin[n]
        else:
            sigma[n] = 0
    data_path = './data/bias_%s/' % bias
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.savetxt(data_path + 'sigma_%s.txt' % bias, (range(N), sigma))


pool = mp.Pool(processes=len(biases))
for bias in biases:
    pool.apply_async(sigma, args=(bias,))
pool.close()
pool.join()

plotar.sigma(bias=biases)

elapsed_sec = timeit.default_timer() - start_time
minutos, segundos = divmod(elapsed_sec, 60)
horas, minutos = divmod(minutos, 60)
print "%d:%02d:%02d\n" % (horas, minutos, segundos)
