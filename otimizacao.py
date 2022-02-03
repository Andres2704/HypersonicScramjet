import fluids.atmosphere as atm
import pandas as pd
from sympy.solvers import nsolve
from sympy import Symbol
from functions import *
from sympy import cos, sin, tan, asin

def otimizacao(n, T_comb, M_ex_comb, M0, h, erro):
    '''
    Código para otimizar os dados da rampa com base na temperatura desejada na câmara de combustão
    com uma tolerância aceitável.
    n           -> Número de Rampas
    T_comb      -> Temperatura na câmara de combustão
    M_ex_comb   -> Número de mach de saída da câmara de combustão (deve ser maior que 1.1)
    M0          -> Número de mach do escoamento livre
    h           -> Altitude de voo em km
    '''
    # Inicializando as variáveis de interesse (pressão, temperatura e densidade)
    P = np.zeros(n+2)
    T = np.zeros(n+2)
    rho = np.zeros(n+2)
    M = np.zeros(n+2)
    beta_i = np.zeros(n+1)
    theta = np.zeros(n)

    # Propriedades atmosféricas em determinada altitude seguindo o padrão 1976
    atmos = atm.ATMOSPHERE_1976(h*1000)

    # Erro calculado
    err = 1

    # Chute inicial do ângulo da primeira rampa
    theta_init = 5*np.pi/180

    # Ajuste do ângulo da rampa para cada iteração
    ajuste = 1*np.pi/180

    while abs(err)>erro:
        # Pressão atmosférica [Pa]
        P[0] = atmos.P
        # Temperatura local
        T[0] = atmos.T
        # Densidade local
        rho[0] = atmos.rho
        # Mach do escoamento livre
        M[0] = M0
        # Ângulo da primeira rampa - chute preliminar
        theta[0] = theta_init

        # Resolvendo as equações analíticas para as râmpas de ondas oblíquas
        for i in range(n+1):
            # Variável para resolver o beta a i-ésima rampa
            beta = Symbol('beta')
            theta_temp = Symbol('theta_temp')
            # O último estágio é deixado para a onda de choque refletido
            if i == n:
                # Equação para o beta
                eqn = tan(sum(theta)) - ((2/tan(beta))*(M[i]**2*sin(beta)**2 - 1)/(M[i]**2*(k + cos(2*beta))+2))
                print('sum_theta: ', sum(theta))
                beta_i[i] = nsolve(eqn, sum(theta))

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

            if i == 0:
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

            else:
                beta_i[i] = asin(M[i-1]*sin(beta_i[i-1])/M[i])

                # Equação para o theta
                eqn = tan(theta_temp) - ((2/tan(beta_i[i]))*(M[i]**2*sin(beta_i[i])**2 - 1)/(M[i]**2*(k + cos(2*beta_i[i]))+2))
                theta[i] = nsolve(eqn, beta_i[i])

                # Mach após a onda de choque oblíqua
                arg = float((1 + (k-1)/2*(M[i]*sin(beta_i[i]))**2)/(k*(M[i]*sin(beta_i[i]))**2 - (k-1)/2))
                M[i+1] = np.sqrt(arg)*(1/sin(beta_i[i] - theta[i]))

                # Pressão após a onda de choque
                P[i+1] = P[i]*(1 + (2*k/(k+1))*((M[i]*sin(beta_i[i]))**2-1))

                # Densidade após a onda de choque
                rho[i+1] = rho[i]*(M[i]*sin(beta_i[i]))**2*(k+1)/(2+(k-1)*(M[i]*sin(beta_i[i]))**2)

                # Temperatura após a onda de choque
                T[i+1] = T[i]*P[i+1]*rho[i]/(P[i]*rho[i+1])

        # Análise do erro
        err = (T[i+1] - T_comb)/T_comb

        if err>0:
            theta_init = theta_init + ajuste/2

        if err<0:
            theta_init = theta_init - ajuste/2

        print(theta_init*180/np.pi)
        print(theta*180/np.pi)

    return np.array([theta, beta, P, T, rho, M])

if __name__ == '__main__':
    # Inputs - Essas entradas com subíndice 0 indicam as propriedades de entrada
    # E de escoamento livre, bem com as informações da primeria rampa do motor
    # Constantes do hidrogênio e o ar
    h_pr = 119.954E6 # poder calorífico j/kg
    f_st = 0.0291 # Fuel/air estequiométrico
    cp = 1012 # Calor específico a pressão cte do ar [j/kgK]
    k = 1.4 # Razão de calores específicos

    # Número de rampas
    n = 2

    # Temperatuta na câmara de combustão em K
    T_comb = 1000

    # Mach do veículo
    M0 = 9

    # Número de mach na saída da câmara de combustão
    M_ex_comb = 1.5

    # Altitude de operação do veículo [km]
    h = 35

    # Erro desejado entre a temperatura na câmara ótima e a calculada
    erro = 1E-4

    scramjet = otimizacao(n, T_comb, M_ex_comb, M0, h, erro)
