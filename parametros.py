#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib; matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'sans-serif', 'size': 14, 'weight': 'normal'}
plt.rc('font', **font)

# Geometria:
H = 2.0         # m
R_int = 0.6     # m
R_ext = 1.6     # m
a = np.sqrt(R_ext**2 - R_int**2)
b = np.sqrt(R_int**2 / a**2)
I = (1.1**2 - R_int**2) / a**2

# Parametros constantes do helimak
f = 1000.  # Hz
v_ph = 2000.  # m/s
w_0 = 2 * np.pi * f
T = 2 * np.pi / w_0
k_z = w_0 / v_ph
v_par = 2000.  # m/s

# Modos
M = 2.
L = round(k_z)

# Amplitude da perturbacao do campo eletrico
phi = -10

# Campo magnetico
B_phi = 1.0  # T
B_z = 0.1 * B_phi

# Fator de seguranca
R_0 = 1.1  # m
R_01 = 1.15  # m
R_02 = 1.25  # m
q_1 = (H * B_phi) / (2 * np.pi * B_z)  # q(r) = q_1 / r
q = q_1 / R_0  # esse q e constante

# Parametros do mapa para q constante
alpha_1 = (T * L) / (q * a * B_z)
alpha = alpha_1 * 2 * phi / a
beta_1 = v_par * B_z * (((M / L) * q) - 1)
#beta = alpha_1 * (beta_1 - E)


def fit_E_vs_r(bias, **kwargs):
    u''' fita campo elétrico em função do raio.
    Dados da planilha do Dennis / Gentle. '''
    data_path = './velocidades.txt'
    grau = kwargs.get('grau', 2)  # ordem do polinomio para fitar E(r)
    ri = kwargs.get('ri', 0.6)  # intervalo dos dados para os quais
    rf = kwargs.get('rf', 1.6)  # fitar uma reta de E vs raio
    biases = np.loadtxt(data_path, usecols=[0], skiprows=1)
    v_data = np.loadtxt(data_path, skiprows=1)
    r_data = np.loadtxt(data_path)[0][1:]
    r_1 = min(enumerate(r_data), key=lambda x: abs(x[1] - ri))  # retorna (indice, valor)
    r_2 = min(enumerate(r_data), key=lambda x: abs(x[1] - rf))  # mais prox. do especificado
    for line in v_data:
        if bias not in biases:
            E_poly = 'Os valores de bias permitidos sao: %s' % biases
        elif line[0] == bias:
            E_data = [ll * 0.077 / rr for ll, rr in zip(line[1:], r_data)]
            E_fit = np.polyfit(r_data[r_1[0]:r_2[0] + 1],
                               E_data[r_1[0]:r_2[0] + 1], grau)
            E_poly = np.poly1d(E_fit)
    return E_poly


def media_E(bias, **kwargs):
    u''' calcula o campo elétrico medio em uma regiao.
    Dados da planilha do Dennis / Gentle. '''
    data_path = './velocidades.txt'
    ri = kwargs.get('ri', 1.15)  # intervalo dos dados para os quais
    rf = kwargs.get('rf', 1.25)  # fitar uma reta de E vs raio
    biases = np.loadtxt(data_path, usecols=[0], skiprows=1)
    v_data = np.loadtxt(data_path, skiprows=1)
    r_data = np.loadtxt(data_path)[0][1:]
    r_1 = min(enumerate(r_data), key=lambda x: abs(x[1] - ri))  # retorna (indice, valor)
    r_2 = min(enumerate(r_data), key=lambda x: abs(x[1] - rf))  # mais prox. do especificado
    for line in v_data:
        if bias not in biases:
            print 'Os valores de bias permitidos sao: %s' % biases
            pass
        elif line[0] == bias:
            E_data = [ll * 0.077 / rr for ll, rr in zip(line[1:], r_data)]
            E_med = np.mean(E_data[r_1[0]:r_2[0]])
    return E_med