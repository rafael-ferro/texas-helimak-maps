#!/usr/bin/python
# -*- coding: UTF-8 -*-

''' PLOTA MAPA NORMALIZADO '''

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
    biases = [float(sys.argv[1])]
else:
    biases = np.loadtxt('velocidades.txt', usecols=[0], skiprows=1)
    biases = [float(bias) for bias in biases]
    
    biases = [7.0,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9,
              8.0,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,  8.8,  8.9,
              9.0,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,  9.9,  10.0]

    
#    biases = np.arange(7.0, 10.1, 0.1)

N = int(1e4)
N_I0 = int(1e4)
X0_w = 0.5

print '\n------------------------------------------------------------'
print 'Mapa I vs Chi'
print 'Numero de iteracoes:', N
print 'Numero de condicoes iniciais:', N_I0


def iteraMapaSemMod(bias, I0, X0, N):
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
            r = np.sqrt(((par.R_ext**2 - par.R_int**2) * I[n + 1])
                                                                + par.R_int**2)
            beta = par.alpha_1 * (par.beta_1 - E(r))
            X[n + 1] = X[n] + beta / np.sqrt(I[n + 1] + par.b**2)
            n += 1
    return X[:n]


def rotacao(bias):
    I0 = np.linspace(0.35, 0.5, N_I0)
    w  = np.zeros(N_I0, float)

    pbar = ProgressBar(widgets=['bias = %s: ' % bias, Percentage(), ' ',
                                Bar(marker='=', left='[', right=']'),
                                ' ', ETA()], term_width=60)
                                
    for i in pbar(range(N_I0)):
        X = iteraMapaSemMod(bias, I0[i], X0_w, N)
        
        if len(X) < N-1:
            # calcula numero de rotacao para cada condicao inicial
            w[i] = (X[-1] - X0_w) / len(X)
        else:
            w[i] = sum([(X[n] - X0_w) / (N+n) for n in range(-1,-51,-1)]) / 50

    data_path = './data/bias_%s/' % bias
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.savetxt(data_path + 'rotacao_%s.txt' % bias, (I0, w))
    

pool = mp.Pool(processes=len(biases))
for bias in biases:
    pool.apply_async(rotacao, args=(bias,))
pool.close()
pool.join()

plotar.rotacao(bias=biases)

elapsed_sec = timeit.default_timer() - start_time
minutos, segundos = divmod(elapsed_sec, 60)
horas, minutos = divmod(minutos, 60)
print "%d:%02d:%02d\n" % (horas, minutos, segundos)
