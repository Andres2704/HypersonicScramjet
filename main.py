import fluids.atmosphere as atm
import numpy as np
import pandas as pd
from sympy.solvers import nsolve
from sympy import Symbol
from functions import *
from sympy import cos, sin, tan

# Inputs - Essas entradas com subíndice 0 indicam as propriedades de entrada
# E de escoamento livre, bem com as informações da primeria rampa do motor
# Constantes do hidrogênio
h_pr = 119.954E6 # poder calorífico j/kg
f_st = 0.0291 # Fuel/air estequiométrico

# Número de rampas
n = 2

# Ângulo das rampas em ordem (primeira, segunda, ..., n-ésima) [deg]
theta = [13.5*np.pi/180, 21.43883272*np.pi/180]
# Inicializando as variáveis de interesse (pressão, temperatura e densidade)
P = np.zeros(n+2)
T = np.zeros(n+2)
rho = np.zeros(n+2)
M = np.zeros(n+2)
beta_i = np.zeros(n+1) # Ângulo da onda de choque

# Altitude de operação do veículo [km]
altitude = 35
# Mach do veículo
M_0 = 9
# Propriedades atmosféricas em determinada altitude seguindo o padrão 1976
atmos = atm.ATMOSPHERE_1976(altitude*1000)
# Pressão atmosférica [Pa]
P[0] = atmos.P
# Temperatura local
T[0] = atmos.T
# Densidade local
rho[0] = atmos.rho
# Mach do escoamento livre
M[0] = M_0
# Razão de calor específico para o ar
k = 1.4

# Resolvendo as equações analíticas para as râmpas de ondas oblíquas
for i in range(n+1):
    # Variável para resolver o beta a i-ésima rampa
    beta = Symbol('beta')

    # O último estágio é deixado para a onda de choque refletido
    if i == n:
        # Equação para o beta
        eqn = tan(sum(theta)) - ((2/tan(beta))*(M[i]**2*sin(beta)**2 - 1)/(M[i]**2*(k + cos(2*beta))+2))
        beta_i[i] = nsolve(eqn, sum(theta))
        print('sum_th', sum(theta))

        # Mach após a onda de choque oblíqua
        arg = float((1 + (k-1)/2*(M[i]*sin(beta_i[i]))**2)/(k*(M[i]*sin(beta_i[i]))**2 - (k-1)/2))
        M[i+1] = np.sqrt(arg)*(1/sin(beta_i[i] - sum(theta)))

        # Pressão após a onda de choque
        P[i+1] = P[i]*(1 + (2*k/(k+1))*((M[i]*sin(beta_i[i]))**2-1))

        # Densidade após a onda de choque
        rho[i+1] = rho[i]*(M[i]*sin(beta_i[i]))**2*(k+1)/(2+(k-1)*(M[i]*sin(beta_i[i]))**2)

        # Temperatura após a onda de choque
        T[i+1] = T[i]*P[i+1]*rho[i]/(P[i]*rho[i+1])

        break


    # Equação para o beta
    eqn = tan(theta[i]) - ((2/tan(beta))*(M[i]**2*sin(beta)**2 - 1)/(M[i]**2*(k + cos(2*beta))+2))
    beta_i[i] = nsolve(eqn, theta[i])

    # Mach após a onda de choque oblíqua
    arg = float((1 + (k-1)/2*(M[i]*sin(beta_i[i]))**2)/(k*(M[i]*sin(beta_i[i]))**2 - (k-1)/2))
    M[i+1] = np.sqrt(arg)*(1/sin(beta_i[i] - theta[i]))

    # Pressão após a onda de choque
    P[i+1] = P[i]*(1 + (2*k/(k+1))*((M[i]*sin(beta_i[i]))**2-1))

    # Densidade após a onda de choque
    rho[i+1] = rho[i]*(M[i]*sin(beta_i[i]))**2*(k+1)/(2+(k-1)*(M[i]*sin(beta_i[i]))**2)

    # Temperatura após a onda de choque
    T[i+1] = T[i]*P[i+1]*rho[i]/(P[i]*rho[i+1])


print('beta: ', beta_i*180/np.pi)
print('Mach: ', M)
print('P: ', P)
print('rho: ', rho)
print('T: ', T)
