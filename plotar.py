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

#font = {'family': 'sans-serif', 'weight': 'normal'}
#plt.rc('font', **font)
font = {'family': 'sans-serif', 'size': 22, 'weight': 'normal'}
dpi = 200

biases_total = np.loadtxt('velocidades.txt', usecols=[0], skiprows=1)
biases_total = [float(bias) for bias in biases_total]
biases_inventados = np.concatenate((np.arange(7,8,0.1),np.arange(8.1,10,0.1)))
biases_data = [bias for bias in biases_total if bias not in biases_inventados]

cores = [(0, 'k')]
iter_cores = iter(plt.cm.rainbow(
                np.linspace(0, 1, sum([b > 0 for b in biases_data]))))
for bias in biases_data:
    if bias > 0:
        cores.append((bias, next(iter_cores)))
iter_cores = iter(plt.cm.seismic_r(
                np.linspace(0, 1, sum([b > 0 for b in biases_inventados]))))
for bias in biases_inventados:
    if bias > 0:
        cores.append((bias, next(iter_cores)))
        

def rotacao(**kwargs):
    u"""Plota numero de rotacao.
     bias = lista (list) de bias para plotar numa mesma figura.
    """
    biases = kwargs.get('bias', biases_total)

    for bias in biases:
        try:
            I, w = np.loadtxt('./data/bias_%s/rotacao_%s.txt' % (bias, bias))
            plt.figure(1, figsize=(5, 5))
            plt.rc('font', **font)
            plt.xlabel(r'$I_0$', fontsize=26)
            plt.ylabel(r'$\omega$', fontsize=26)
            plt.tick_params(length=8, width=1)
            plt.gca().ticklabel_format(useOffset=False)
            plt.plot(I, w, '.k', markersize=1)
#            plt.locator_params(axis = 'y', nbins = 8)
#            plt.locator_params(axis = 'x', nbins = 6)

            if bias == 0:
                plt.ylim(1.084, 1.092)
                plt.xlim(0.40, 0.45)
                plt.locator_params(axis = 'y', nbins = 8)
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.410))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.415))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_max = np.argmax(w_shear)
                I_max = I_shear[w_max]
                plt.plot(I_max, w_shear[w_max], 'ob', markersize=5)
            elif bias == 4:
                plt.ylim(1.26, 1.29)
                plt.xlim(0.4, 0.5)
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.43))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.44))
                I_shear = I[I1:I2]      
                w_shear = w[I1:I2]
                w_min = np.argmin(w_shear)
                I_min = I_shear[w_min]
                plt.plot(I_min, w_shear[w_min], 'ob', markersize=5)
            elif bias == 8.4:
                plt.locator_params(axis = 'y', nbins = 8)
                plt.ylim(1.172, 1.176)
                plt.xlim(0.36, 0.40)
            else:
                plt.ylim(1.166, 1.178)
                plt.xlim(0.37, 0.47)

            plt.savefig('./figs/rotacao_%s.png' % bias, bbox_inches='tight', dpi=dpi)
            print 'Imagem salva como ./figs/rotacao_%s.png' % bias
            plt.close()
        except IOError:
            pass


def inset_rotacao(**kwargs):
    u"""Plota numero de rotacao.
     bias = lista (list) de bias para plotar numa mesma figura.
    """
    biases = kwargs.get('bias', biases_total)

    for bias in biases:
        try:
            I, w = np.loadtxt('./data/bias_%s/rotacao_%s.txt' % (bias, bias))
            fig, ax1 = plt.subplots(1, figsize=(5, 5))
            plt.rc('font', **font)
            plt.xlabel(r'$I_0$', fontsize=26)
            plt.ylabel(r'$\omega$', fontsize=26)
            plt.gca().ticklabel_format(useOffset=False)
            ax1.plot(I, w, '.k', markersize=0.5)
            ax1.tick_params(length=8, width=1)
            
            if bias == 7.8:
                plt.ylim(1.166, 1.178)
                plt.xlim(0.37, 0.47)
                
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.43))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.44))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmin(w_shear)

                i1 = 0.4345
                i2 = 0.4405
                x1 = 1.16895
                x2 = 1.16905
                
                ax2 = fig.add_axes([0.62, 0.63, 0.2, 0.2])
            
            elif bias == 8.5:
                plt.ylim(1.172, 1.176)
                plt.xlim(0.36, 0.40)
                ax1.locator_params(axis = 'y', nbins = 6)
                ax1.locator_params(axis = 'x', nbins = 6)
            
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.368))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.370))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmax(w_shear)
                I_m = I_shear[w_m]

                i1 = 0.364
                i2 = 0.372
                x1 = 1.17550
                x2 = 1.17555

                ax2 = fig.add_axes([0.35, 0.2, 0.2, 0.2])
                ax2.plot(I, w, '.k', markersize=1)
                ax2.plot(I_m, w_shear[w_m], 'ob', markersize=5)
                
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.365))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.367))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmin(w_shear)
                
            elif bias == 8.6:
                plt.ylim(1.172, 1.176)
                plt.xlim(0.36, 0.40)
                ax1.locator_params(axis = 'y', nbins = 6)
                ax1.locator_params(axis = 'x', nbins = 6)
            
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.368))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.370))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmax(w_shear)

                i1 = 0.362
                i2 = 0.372
                x1 = 1.1755
                x2 = 1.1758

                ax2 = fig.add_axes([0.35, 0.2, 0.2, 0.2])

            elif bias == 8.7:
                plt.ylim(1.172, 1.176)
                plt.xlim(0.36, 0.40)
                ax1.locator_params(axis = 'y', nbins = 6)
                ax1.locator_params(axis = 'x', nbins = 6)
            
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.368))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.370))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmax(w_shear)

                i1 = 0.362
                i2 = 0.372
                x1 = 1.1755
                x2 = 1.1759

                ax2 = fig.add_axes([0.35, 0.2, 0.2, 0.2])
                
            elif bias == 8.8:
                plt.ylim(1.172, 1.176)
                plt.xlim(0.36, 0.40)
                ax1.locator_params(axis = 'y', nbins = 6)
                ax1.locator_params(axis = 'x', nbins = 6)
            
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.368))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.370))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmax(w_shear)

                i1 = 0.362
                i2 = 0.372
                x1 = 1.17561
                x2 = 1.17595

                ax2 = fig.add_axes([0.35, 0.2, 0.2, 0.2])

            elif bias == 8.9:
                plt.ylim(1.172, 1.176)
                plt.xlim(0.36, 0.40)
                ax1.locator_params(axis = 'y', nbins = 6)
                ax1.locator_params(axis = 'x', nbins = 6)
            
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.368))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.370))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmax(w_shear)

                i1 = 0.362
                i2 = 0.372
                x1 = 1.17561
                x2 = 1.17595

                ax2 = fig.add_axes([0.35, 0.2, 0.2, 0.2])
                
            elif bias == 9:
                plt.ylim(1.172, 1.176)
                plt.xlim(0.36, 0.40)
                ax1.locator_params(axis = 'y', nbins = 6)
                ax1.locator_params(axis = 'x', nbins = 6)
            
                I1 = min(range(len(I)), key=lambda i: abs(I[i]-0.368))
                I2 = min(range(len(I)), key=lambda i: abs(I[i]-0.370))
                I_shear = I[I1:I2]
                w_shear = w[I1:I2]
                w_m = np.argmax(w_shear)

                i1 = 0.362
                i2 = 0.372
                x1 = 1.17561
                x2 = 1.17595

                ax2 = fig.add_axes([0.35, 0.2, 0.2, 0.2])

            I_m = I_shear[w_m]
            ax1.plot(I_m, w_shear[w_m], 'ob', markersize=1)
            ax1.add_patch(Rectangle((i1, x1), i2-i1, x2-x1,
                                      facecolor='none', edgecolor='r',
                                      linewidth=0.75))
            
            ax2.set_ylim(x1, x2)
            ax2.set_xlim(i1, i2)
            ax2.tick_params(labelsize=16)

            ax2.plot(I, w, '.k', markersize=1)
            ax2.plot(I_m, w_shear[w_m], 'ob', markersize=5)
            
            ax2.set_xticks([i1, i2])
            ax2.set_xticklabels([i1, i2])
            ax2.set_yticks([x1, x2])
            ax2.set_yticklabels([x1, x2])
            
            ax2.spines['bottom'].set_color('red')
            ax2.spines['top'].set_color('red') 
            ax2.spines['right'].set_color('red')
            ax2.spines['left'].set_color('red')
            
            plt.savefig('./figs/inset_rotacao_%s.png' % bias, bbox_inches='tight', dpi=dpi)
            print 'Imagem salva como ./figs/inset_rotacao_%s.png' % bias
            plt.close()
        except IOError:
            pass
        

def rotacoes(**kwargs):
    u"""Plota numero de rotacao.
     bias = lista (list) de bias para plotar numa mesma figura.
    """
    biases = kwargs.get('bias', biases_total)
    
    # norm is a class which, when called, can normalize data into the
    # [0.0, 1.0] interval.
    norm = matplotlib.colors.Normalize(vmin=min(biases), vmax=max(biases))
    
    # choose a colormap
    c_m = matplotlib.cm.Set1
    
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    plt.figure(1)#, figsize=(6, 6))
    plt.xlabel(r'$I_0$')
    plt.ylabel(r'$\omega$')
    plt.ylim(1.168, 1.176)
    plt.xlim(0.360, 0.415)
#    plt.locator_params(axis = 'y', nbins = 6)
    plt.locator_params(axis = 'x', nbins = 6)
    plt.gca().ticklabel_format(useOffset=False)
            
    for bias in biases:
        try:
            I, w = np.loadtxt('./data/bias_%s/rotacao_%s.txt' % (bias, bias))

            plt.plot(I, w, '.', markersize=1, color=s_m.to_rgba(bias))
            
        except IOError:
            pass
    
    # having plotted the 11 curves we plot the colorbar, using again our
    # ScalarMappable
    cbar = plt.colorbar(s_m)
    cbar.set_label('$ bias $', rotation=270, labelpad=20)
#    cbar.set_ticks(biases)
#    cbar.set_ticklabels(biases)
    
    plt.savefig('./figs/rotacoes.png', bbox_inches='tight', dpi=dpi)
    print 'Imagem salva como ./figs/rotacoes.png'
    plt.close()


def rotacao_inflex(**kwargs):
    u"""Plota numero de rotacao.
     bias = lista (list) de bias para plotar numa mesma figura.
    """
    biases = kwargs.get('bias', biases_total)

    for bias in biases:
        try:
            I, w = np.loadtxt('./data/bias_%s/rotacao_%s.txt' % (bias, bias))
            sortId=np.argsort(I)
            I=I[sortId]
            w=w[sortId]
            maxm = argrelextrema(w, np.greater)
            minm = argrelextrema(w, np.less)
            plt.figure(1, figsize=(5, 4))
            plt.xlabel(r'$I_0$')
            plt.ylabel(r'$\omega$')
            ymin = min(w) - ((max(w) - min(w))/100)
            ymax = max(w) + ((max(w) - min(w))/100)
            plt.ylim(ymin,ymax)
            plt.ylim(1.172, 1.176)
            plt.xlim(0.36, 0.40)
            plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
            plt.plot(I, w, '.k', markersize=1)
            plt.plot(I[maxm], w[maxm], 'og', markersize=3)
            plt.plot(I[minm], w[minm], 'ob', markersize=3)
#            plt.plot(I[np.argmax(w)], max(w), 'om', markersize=3)
#            plt.savefig('./figs/rotacao2_%s.png' 
#                        % bias, bbox_inches='tight', dpi=dpi)
#            print 'Imagem salva como ./figs/rotacao2_%s.png' % bias
            plt.show()
        except IOError:
            pass


def E_vs_r(**kwargs):
    u"""Plota E(r) fitado dos dados de Gentle.
     bias = lista (list) de bias para plotar numa mesma figura
     ri = raio inicial :: rf = raio final  para fitar o polinomio
     grau = grau do polinomio a ser fitado (padrao 1)
    """
    biases = kwargs.get('bias', biases_data)
    ri = kwargs.get('ri', 0.6)  # intervalo dos dados para os quais
    rf = kwargs.get('rf', 1.6)  # fitar uma reta de E vs raio
    grau = kwargs.get('grau', 2)  # ordem do polinomio para fitar E(r)
    v_data = np.loadtxt('velocidades.txt', skiprows=1)
    r_data = np.loadtxt('velocidades.txt')[0][1:]
    r_1 = min(enumerate(r_data), key=lambda x: abs(x[1] - ri))  # retorna (indice, valor)
    r_2 = min(enumerate(r_data), key=lambda x: abs(x[1] - rf))  # mais prox. do especificado
    ri_fit = 0.9
    rf_fit = 1.3
    r = np.linspace(ri_fit, rf_fit, 50)
    
    # norm is a class which, when called, can normalize data into the
    # [0.0, 1.0] interval.
    norm = matplotlib.colors.Normalize(vmin=min(biases), vmax=max(biases))
    
    # choose a colormap
    c_m = matplotlib.cm.Greys
    
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    plt.figure(1, figsize=(8,6))
    plt.rcParams.update({'legend.fontsize': 24, 'font.size': 24})
    plt.xlabel('r (m)')
    plt.ylabel(u'electric field (V/m)')
    plt.xlim(ri_fit, rf_fit)
    plt.xticks([0.9, 1.0, 1.1, 1.2, 1.3])
    plt.yticks([-50, 0, 50, 100])
#    plt.locator_params(axis = 'y', nbins = 5)

    for bias in biases:
        for line in v_data:
            if bias not in biases:
                print "Os valores de bias permitidos sao: %s" % biases_total
            elif line[0] == bias:
                E_data = [ll * 0.077 / rr for ll, rr in zip(line[1:], r_data)]
#                E_fit = np.polyfit(r_data[r_1[0]:r_2[0] + 1], E_data[r_1[0]:r_2[0] + 1], grau)
#                E_poly = np.poly1d(E_fit)
#                E_r = E_poly(r)

                plt.plot(r_data, E_data, '-', lw=1, color=s_m.to_rgba(bias))
                
#                plt.plot([min(r), max(r)], [par.beta_1, par.beta_1], '-k', lw=.5)
#                plt.plot([min(r), max(r)], [0, 0], '-.', color='0.75', lw=.1)
#                plt.text(1.15, 170,
#                         r'$\beta_1 = \frac{v_\parallel B_z}{L} \left(M q - L \right)$', fontsize=10)

    cbar = plt.colorbar(s_m)
    cbar.set_label('$ bias $ (V)')#, rotation=270, labelpad=20)
#    cbar.set_ticks(biases)
#    cbar.set_ticklabels(biases)
    
    plt.savefig('./figs/E_vs_r.png', bbox_inches='tight', dpi=dpi)
    print 'Imagem salva como ./figs/E_vs_r.png'
    plt.close()


def SE_vs_r(**kwargs):
    u"""Plota E(r) fitado dos dados de Gentle.
     bias = lista (list) de bias para plotar numa mesma figura
     ri = raio inicial :: rf = raio final  para fitar o polinomio
     grau = grau do polinomio a ser fitado (padrao 1)
    """
    biases = kwargs.get('bias', biases_total)
    ri = kwargs.get('ri', 0.6) # intervalo dos dados para os quais
    rf = kwargs.get('rf', 1.6) # fitar uma reta de E vs raio
    grau = kwargs.get('grau', 2) # ordem do polinomio para fitar E(r)
    v_data = np.loadtxt('velocidades.txt', skiprows = 1)
    r_data = np.loadtxt('velocidades.txt')[0][1:]
    r_1 = min(enumerate(r_data), key=lambda x:abs(x[1]-ri)) #retorna (indice, valor)
    r_2 = min(enumerate(r_data), key=lambda x:abs(x[1]-rf)) #mais prox. do especificado
    ri_fit = 0.9
    rf_fit = 1.3
    raio = np.linspace(ri_fit, rf_fit, 100)
    plt.figure(1, figsize=(5, 5))
    plt.xlabel('raio (m)')
    plt.ylabel(r"$S_{E_r}$")
    plt.xlim(ri_fit, rf_fit)
    plt.xticks([0.9, 1.0, 1.1, 1.2, 1.3])
    plt.ylim(-1500,1500)
#    plt.locator_params(axis = 'y', nbins = 4)
    
    for bias in biases:
        for line in v_data:
            if bias not in biases:
                print "Os valores de bias permitidos sao: %s" %biases_total
            elif line[0] == bias:
                E_data = [ll * 0.077 / rr for ll, rr in zip(line[1:], r_data)]
                E_fit = np.polyfit(r_data[r_1[0]:r_2[0]+1], E_data[r_1[0]:r_2[0]+1], grau)
                E_poly = np.poly1d(E_fit)
                dE_dr = E_poly.deriv()
                SEr = dE_dr(raio)
                cor = [c[1] for c in cores if c[0] == bias][0]
                plt.plot(raio,SEr, '-', color = cor, label = int(bias))
                                    
    plt.legend(title='bias(V)', loc=0, bbox_to_anchor=(1.01,1.02))
    plt.savefig('./figs/SE_vs_r.png', bbox_inches='tight', dpi=dpi)
    print 'Imagem salva como ./figs/SE_vs_r.png'
    plt.close()


def beta_vs_r(**kwargs):
    u"""Plota beta(r) fitado dos dados de Gentle.
     bias = lista (list) de bias para plotar numa mesma figura
     ri = raio inicial :: rf = raio final  para fitar o polinomio
     grau = grau do polinomio a ser fitado (padrao 1)
    """
    biases = kwargs.get('bias', biases_total)
    ri = kwargs.get('ri', 0.6)  # intervalo dos dados para os quais
    rf = kwargs.get('rf', 1.6)  # fitar uma reta de E vs raio
    grau = kwargs.get('grau', 2)  # ordem do polinomio para fitar E(r)
    v_data = np.loadtxt('velocidades.txt', skiprows=1)
    r_data = np.loadtxt('velocidades.txt')[0][1:]
    r_1 = min(enumerate(r_data), key=lambda x: abs(x[1] - ri))  # retorna (indice, valor)
    r_2 = min(enumerate(r_data), key=lambda x: abs(x[1] - rf))  # mais prox. do especificado
    ri_fit = 0.9
    rf_fit = 1.3
    r = np.linspace(ri_fit, rf_fit, 50)
    plt.figure(1, figsize=(5, 5))
    plt.xlabel('raio (m)')
    plt.ylabel(r'$\beta$')
#    plt.xlim(ri_fit, rf_fit)
    plt.xlim(1.1, 1.25)
    plt.ylim(0.8, 1.0)

    for bias in biases:
        for line in v_data:
            if bias not in biases:
                print "Os valores de bias permitidos sao: %s" % biases_total
            elif line[0] == bias:
                E_data = [ll * 0.077 / rr for ll, rr in zip(line[1:], r_data)]
                E_fit = np.polyfit(r_data[r_1[0]:r_2[0] + 1], E_data[r_1[0]:r_2[0] + 1], grau)
                E_r = np.poly1d(E_fit, variable='r')
                beta = float(par.alpha_1) * (par.beta_1 - E_r)
                print beta
                crit = beta.deriv().r
                r_crit = crit[crit.imag==0].real
                test = beta.deriv(2)(r_crit)
                x_min = r_crit[test>0]
                y_min = beta(x_min)
                cor = [c[1] for c in cores if c[0] == bias][0]
                plt.plot(r, beta(r), '-', color=cor, label=int(bias))
                plt.plot(x_min, y_min, 'o', color=cor, markersize=3)
                
    plt.legend(title='bias(V)', loc=0, fontsize=10, bbox_to_anchor=(1.01,1.02))
    plt.savefig('./figs/beta_vs_r.png', bbox_inches='tight', dpi=dpi)
    print 'Imagem salva como ./figs/beta_vs_r.png'
    plt.close()


def fit_omega_vs_I(**kwargs):
    u"""Plota omega(I) fitado dos dados de Gentle.
     bias = lista (list) de bias para plotar numa mesma figura
     ri = raio inicial :: rf = raio final  para fitar o polinomio
     grau = grau do polinomio a ser fitado (padrao 1)
    """
    biases = kwargs.get('bias', biases_total)
    ri = kwargs.get('ri', 0.6)  # intervalo dos dados para os quais
    rf = kwargs.get('rf', 1.6)  # fitar uma reta de E vs raio
    grau = kwargs.get('grau', 2)  # ordem do polinomio para fitar E(r)
    v_data = np.loadtxt('velocidades.txt', skiprows=1)
    r_data = np.loadtxt('velocidades.txt')[0][1:]
    r_1 = min(enumerate(r_data), key=lambda x: abs(x[1] - ri))  # retorna (indice, valor)
    r_2 = min(enumerate(r_data), key=lambda x: abs(x[1] - rf))  # mais prox. do especificado
    I_data = (r_data**2 - par.R_int**2) / (par.R_ext**2 - par.R_int**2)
#    I_1 = (r_1[1]**2 - par.R_int**2) / (par.R_ext**2 - par.R_int**2)
#    I_2 = (r_2[1]**2 - par.R_int**2) / (par.R_ext**2 - par.R_int**2)
    I_reta = np.linspace(0.4, 0.5, 10000)

    plt.figure(1, figsize=(5, 5))
    plt.xlabel(r'$I$')
    plt.ylabel(r'$\omega \quad (\Delta \chi)$')
    plt.xlim(min(I_reta), max(I_reta))

    for bias in biases:
        for line in v_data:
            if bias not in biases:
                print "Os valores de bias permitidos sao: %s" % biases_total
            elif line[0] == bias:
                E_data = [ll * 0.077 / ii for ll, ii in zip(line[1:], I_data)]
                E_fit = np.polyfit(I_data[r_1[0]:r_2[0] + 1], E_data[r_1[0]:r_2[0] + 1], grau)
                E_I = np.poly1d(E_fit, variable='I')
                beta = np.poly1d(par.alpha_1 * (par.beta_1 - E_I), variable='I')
                print beta
                omega = beta(I_reta)/np.sqrt(I_reta**2 + par.b**2)
                omega_fit = np.polyfit(I_reta,omega, 2)
                omega = np.poly1d(omega_fit, variable='I')
#                print omega
                crit = omega.deriv().r
                r_crit = crit[crit.imag==0].real
                test = omega.deriv(2)(r_crit)
                x_min = r_crit[test>0]
                y_min = omega(x_min)
#                print "xmin = %s, ymin = %s\n\n" %(x_min,y_min)
                cor = [c[1] for c in cores if c[0] == bias][0]
                plt.plot(I_reta, omega(I_reta), '-', color=cor, label=float(bias))
                plt.plot(x_min, y_min, 'o', color=cor, markersize=5)
#                plt.annotate('(%s, %s)' % (round(x_min,2),round(y_min,2)),
#                             xy=(x_min,y_min), ha='center', va='top', fontsize=8)
                
    plt.legend(title='bias(V)', loc=0, bbox_to_anchor=(1.01,1.02))
    plt.savefig('./figs/fit_omega_vs_I.png', bbox_inches='tight', dpi=dpi)
    print 'Imagem salva como ./figs/fit_omega_vs_I.png'
    plt.close()


def omega_vs_I(**kwargs):
    u"""Plota omega(I) fitado dos dados de Gentle.
     bias = lista (list) de bias para plotar numa mesma figura
     ri = raio inicial :: rf = raio final  para fitar o polinomio
     grau = grau do polinomio a ser fitado (padrao 1)
    """
    biases = kwargs.get('bias', biases_total)
    ri = kwargs.get('ri', 0.6)  # intervalo dos dados para os quais
    rf = kwargs.get('rf', 1.6)  # fitar uma reta de E vs raio
    grau = kwargs.get('grau', 2)  # ordem do polinomio para fitar E(r)
    v_data = np.loadtxt('velocidades.txt', skiprows=1)
    r_data = np.loadtxt('velocidades.txt')[0][1:]
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

    plt.figure(1, figsize=(8,6))
    plt.rcParams.update({'legend.fontsize': 16, 'font.size': 22})
    plt.xlabel(r'$ I $')
    plt.ylabel(r'$ \omega $')
    plt.xlim(min(I), max(I))
#    plt.xlim(0.2, 0.6)
    plt.locator_params(axis = 'x', nbins = 4)

    for bias in biases:
        for line in v_data:
            if bias not in biases:
                print "Os valores de bias permitidos sao: %s" % biases_total
            elif line[0] == bias:
                E_data = [ll * 0.077 / ii for ll, ii in zip(line[1:], I_data)]
                E_fit = np.polyfit(I_data[r_1[0]:r_2[0] + 1], E_data[r_1[0]:r_2[0] + 1], grau)
                E_I = np.poly1d(E_fit, variable='I')
                beta = np.poly1d(par.alpha_1 * (par.beta_1 - E_I), variable='r')
#                print beta
                omega = beta(I)/np.sqrt(I**2 + par.b**2)
#                print omega

#                x_min = I[np.argmin(omega)]
#                y_min = min(omega)
                plt.plot(I, omega, '-', lw=1, color=s_m.to_rgba(bias))
#                plt.plot(x_min, y_min, 'o', color=cor, markersize=3)
#                plt.annotate('(%s, %s)' % (round(x_min,2),round(y_min,2)),
#                        xy=(x_min,y_min), ha='center', va='top', fontsize=8)
                
    cbar = plt.colorbar(s_m)
    cbar.set_label('$ bias $ (V)', rotation=270, labelpad=20)
    
    plt.savefig('./figs/omega_vs_I.png', bbox_inches='tight', dpi=dpi)
    print 'Imagem salva como ./figs/omega_vs_I.png'
    plt.close()