#!/usr/bin/python
# -*- coding: utf-8 -*-
import locale
#locale.setlocale(locale.LC_ALL, 'pt_BR.utf-8')
locale.setlocale(locale.LC_ALL, 'en_US.utf-8')
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from scipy.signal import argrelextrema

import parametros as par

if not os.path.exists('./figs/'):
    os.makedirs('./figs/')

dpi = 200

#biases = np.concatenate((np.arange(7,8,0.1), np.arange(8.1,10,0.1)))
biases = [ 7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9,  8. ,
          8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,  8.8,  8.9,  9. ,  9.1,
          9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,  9.9, 10. ]

v_data = np.loadtxt('velocidades.txt', skiprows=1)
r_data = np.loadtxt('velocidades.txt')[0][1:]

ri = 0.6  # intervalo dos dados para os quais
rf = 1.6  # fitar polinomio de E vs raio
grau = 2  # ordem do polinomio para fitar E(r)

r_1 = min(enumerate(r_data), key=lambda x: abs(x[1] - ri))  # retorna (indice, valor)
r_2 = min(enumerate(r_data), key=lambda x: abs(x[1] - rf))  # mais prox. do especificado

I_data = (r_data**2 - par.R_int**2) / (par.R_ext**2 - par.R_int**2)
I_1 = (r_1[1]**2 - par.R_int**2) / (par.R_ext**2 - par.R_int**2)
I_2 = (r_2[1]**2 - par.R_int**2) / (par.R_ext**2 - par.R_int**2)
I = np.linspace(I_1, I_2, 10000)

# norm is a class which, when called, can normalize data into the
# [0.0, 1.0] interval.
norm = matplotlib.colors.Normalize(vmin=min(biases), vmax=max(biases))

# choose a colormap
c_m = matplotlib.cm.Greys

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

fig = plt.figure(1, figsize=(5,5))
plt.rcParams.update({'legend.fontsize': 12, 'font.size': 18})
plt.xlabel(r'$ I $')
plt.locator_params(axis='x', nbins = 6)

axE = fig.add_subplot(111)
axE.set_ylabel(u'electric field $ E_0 $ (V/m)')
#axE.set_xlim(I_1, I_2)
#axE.xticks([0.9, 1.0, 1.1, 1.2, 1.3])
axo = axE.twinx()
axo.set_ylabel(r'$ \omega $', fontsize = 26)


for line in v_data:
    if line[0] in biases:
        bias = line[0]
        E_data = [ll * 0.077 / rr for ll, rr in zip(line[1:], r_data)]
        E_fit = np.polyfit(I_data[r_1[0]:r_2[0] + 1], 
                           E_data[r_1[0]:r_2[0] + 1], grau)
        E_I = np.poly1d(E_fit, variable='I')
        beta = np.poly1d(par.alpha_1 * (par.beta_1 - E_I), variable='I')            
        omega = beta(I)/np.sqrt(I**2 + par.b**2)
        
        axE.plot(I, E_I(I), '-', lw=1, color=s_m.to_rgba(bias))
        axE.text(0.31, 60, r'$ E_0 $')
        axo.plot(I, omega, '-', lw=1, color=s_m.to_rgba(bias))
        axo.text(0.32, 1.6, r'$ \omega $')
        
    elif line[0] == 0 or line[0] == 4:
        bias = line[0]
        E_data = [ll * 0.077 / rr for ll, rr in zip(line[1:], r_data)]
        E_fit = np.polyfit(I_data[r_1[0]:r_2[0] + 1], 
                           E_data[r_1[0]:r_2[0] + 1], grau)
        E_I = np.poly1d(E_fit, variable='I')
        beta = np.poly1d(par.alpha_1 * (par.beta_1 - E_I), variable='I')            
        omega = beta(I)/np.sqrt(I**2 + par.b**2)
        
        if bias == 0:
            axE.plot(I, E_I(I), '--k', lw=1, label = r'$ bias = 0 $ V')
#            axE.text(0.31, 60, r'$ E_0 (bias = 0 V) $')
            axo.plot(I, omega, '--k', lw=1)
#            axo.text(0.32, 1.6, r'$ \omega (bias = 0 V) $')
        else:
            axE.plot(I, E_I(I), '-.k', lw=1, label = r'$ bias = 4 $ V')
#            axE.text(0.31, 60, r'$ E_0 (bias = 4 V) $')
            axo.plot(I, omega, '-.k', lw=1)
#            axo.text(0.32, 1.6, r'$ \omega (bias = 4 V) $')
        axE.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=14)

cbaxes = fig.add_axes([1.15, 0.1, 0.03, 0.8])
cbticks = [ 7. , 7.5,  8., 8.5, 9., 9.5, 10. ]
cbar = plt.colorbar(s_m, cax=cbaxes)
cbar.set_label('$ bias $ (V)', rotation=90, labelpad=5)
cbar.set_ticks(cbticks)
cbar.set_ticklabels(cbticks)

plt.savefig('./figs/E_omega_vs_I.png', bbox_inches='tight', dpi=dpi)
print 'Imagem salva como ./figs/omega_vs_I.png'
plt.close()