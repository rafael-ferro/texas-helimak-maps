#!/usr/bin/python
# -*- coding: UTF-8 -*-

''' PLOTA MAPA COM CURVA SEM SHEAR '''

import timeit
start_time = timeit.default_timer()

import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import multiprocessing as mp
import numpy as np

import os
import sys

from progressbar import ProgressBar, Percentage, Bar, ETA

#from mapas import mapa_E_vs_r
import parametros as par

if len(sys.argv) > 1:
    biases = [float(sys.argv[1])]
else:
#    biases = np.loadtxt('velocidades.txt', usecols=[0], skiprows=1)
#    biases = [float(bias) for bias in biases]
    
    biases = [7.8]

print '\n------------------------------------------------------------'
print 'Curva Shearless'

def itera_mapa_E_vs_r(bias, I0, X0, N):
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
            r = np.sqrt(((par.R_ext**2 - par.R_int**2) * I[n + 1]) + par.R_int**2)
            beta = par.alpha_1 * (par.beta_1 - E(r))
            X[n + 1] = X[n] + beta / np.sqrt(I[n + 1] + par.b**2)
            n += 1
    return X[:n]


def mapa(bias, Is, Xs):
    font = {'family': 'sans-serif', 'size': 16, 'weight': 'normal'}
    plt.rc('font', **font)
    fig = plt.figure(1, figsize=(5, 5))
    plt.axis([0, 1, -par.b**2, 1])
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$\chi$')
    ax1.set_ylabel('$I$')
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    pbar = ProgressBar(widgets=['mapa (%s): ' % bias, Percentage(), ' ',
                                Bar(marker='=', left='[', right=']'),
                                ' ', ETA()], term_width=60)

    for X0 in pbar(np.linspace(0.0, 1.0, 8)):
        for I0 in np.linspace(0.0, 1.0, 8):
            I, X = itera_mapa_E_vs_r(bias, I0, X0, 800)
            ax1.plot(X, I, '.k', markersize=1)
    
    I, X = itera_mapa_E_vs_r(bias, Is, Xs, int(1e5))
    ax1.plot(X, I, '.b', markersize=1)
    
    ax2 = ax1.twinx()
    ax2.set_yticks([-par.b**2, 0, 1])
    ax2.set_yticklabels(
        ['$\mathrm{-b}^2$', '$\mathrm{R_{int}}$', '$\mathrm{R_{ext}}$'])

    fig_path = './figs/mapas/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    arquivo_py = os.path.basename(__file__)
    fig_nome = 'mapa_' + os.path.splitext(arquivo_py)[0] + '_%s.png' % bias
    plt.savefig(fig_path + fig_nome, bbox_inches='tight', dpi=200)
    print 'Imagem salva como %s%s.png' % (fig_path, fig_nome)
    plt.close()


def rotacao(bias):
    N_I0 = int(1e3)
    N = int(1e4)
    X0_w = 0.5
    I1 = 0.435
    I2 = 0.439
    I0 = np.linspace(I1, I2, N_I0) 
    w = np.zeros(N_I0, float)

    pbar = ProgressBar(widgets=['rotacao (%s): ' % bias, Percentage(), ' ',
                            Bar(marker='=', left='[', right=']'),
                            ' ', ETA()], term_width=60)

    for i in pbar(range(N_I0)):
        X = iteraMapaSemMod(bias, I0[i], X0_w, N)
        if len(X) < N-1:
            # calcula numero de rotacao para cada condicao inicial
            w[i] = (X[-1] - X0_w) / len(X)
        else:
            w[i] = sum([(X[n] - X0_w) / (N+n) for n in range(-1,-51,-1)]) / 50
    I0_w = I0[np.argmin(w)]

    mapa(bias, I0_w, X0_w)
    data_path = './data/bias_%s/' % bias
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.savetxt(data_path + 'shearless_%s.txt' % bias, (I0_w, X0_w))
    

pool = mp.Pool(processes=len(biases))
for bias in biases:
    pool.apply_async(rotacao, args=(bias,))
pool.close()
pool.join()

elapsed_sec = timeit.default_timer() - start_time
minutos, segundos = divmod(elapsed_sec, 60)
horas, minutos = divmod(minutos, 60)
print "%d:%02d:%02d\n" % (horas, minutos, segundos)
