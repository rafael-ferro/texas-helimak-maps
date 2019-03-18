#!/usr/bin/python
# -*- coding: UTF-8 -*-

''' PLOTA MAPA NORMALIZADO '''

import timeit
start_time = timeit.default_timer()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import multiprocessing as mp
import numpy as np

import os
import sys

from progressbar import ProgressBar, Percentage, Bar, ETA

import parametros as par

fig_path = './figs/mapas/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

if len(sys.argv) > 1:
    biases = [float(sys.argv[1])]
else:
    biases = np.loadtxt('velocidades.txt', usecols=[0], skiprows=1)
    biases = [float(bias) for bias in biases]
    
    biases = [7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9,  8. ,
        8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,  8.8,  8.9,  9. ,  9.1,
        9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,  9.9, 10. ]


N = int(1e3)
N_X0 = 10
N_I0 = 10
X0 = np.linspace(0, 1, N_X0)
I0 = np.linspace(0, 1, N_I0)

print '\n------------------------------------------------------------'
print 'Mapa I vs Chi'
print 'Numero de iteracoes:', N
print 'Numero de condicoes iniciais:', N_X0 * N_I0


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


def plota_mapa(bias):
    fig = plt.figure(1, figsize=(5, 5))
    font = {'family': 'sans-serif', 'size': 28, 'weight': 'normal'}
    plt.rc('font', **font)
    plt.axis([0, 1, -par.b**2, 1])
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$\chi$')
    ax1.set_ylabel('$I$')
    plt.title(r'$bias = %s$ V' %bias)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax1.xaxis.set_ticks([0, 1])
    ax1.yaxis.set_ticks([0, 1])

    pbar = ProgressBar(widgets=['bias = %s: ' % bias, Percentage(), ' ',
                                Bar(marker='=', left='[', right=']'),
                                ' ', ETA()], term_width=60)

    for x in pbar(X0):
        for i in I0:
            I, X = itera_mapa_E_vs_r(bias, i, x, N)
            ax1.plot(X, I, '.k', markersize=1)

#    I, w = np.loadtxt('./data/bias_%s/rotacao_%s.txt' % (bias, bias))
#    I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.410))
#    I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.415))
#    I_shear = I[I1:I2]
#    w_shear = w[I1:I2]
#    w_max = np.argmax(w_shear)
#    I_max = I_shear[w_max]
#    w_min = np.argmin(w_shear)
#    I_min = I_shear[w_min]
#    I, X = itera_mapa_E_vs_r(bias, I_max, 0.5, int(1e4))
#    ax1.plot(X, I, '.r', markersize=1)

    ax2 = ax1.twinx()
    ax2.set_yticks([-par.b**2, 0, 1])
    ax2.set_yticklabels(
        ['$\mathrm{-b}^2$', '$\mathrm{R_{int}}$', '$\mathrm{R_{ext}}$'])

    arquivo_py = os.path.basename(__file__)
    fig_nome = os.path.splitext(arquivo_py)[0] + '_%s' % bias
    plt.savefig(fig_path + fig_nome + '.png', bbox_inches='tight', dpi=200)
    print 'Imagem salva como %s%s.png' % (fig_path, fig_nome)
    plt.show()


pool = mp.Pool(processes=len(biases))
for bias in biases:
    pool.apply_async(plota_mapa, args=(bias,))
pool.close()
pool.join()

elapsed_sec = timeit.default_timer() - start_time
minutos, segundos = divmod(elapsed_sec, 60)
horas, minutos = divmod(minutos, 60)
print "%d:%02d:%02d\n" % (horas, minutos, segundos)
