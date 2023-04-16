import fluids.atmosphere as atm
import pandas as pd
from sympy.solvers import nsolve
from sympy import Symbol
from functions import *
from sympy import cos, sin, tan, asin

def optimize(n, T_comb, M0, h, erro, incremento, theta_init):
    '''
    Code constructed for optimizate the ramps in a Scramjet based on a desired combustion chamber temperature.

    ==> Inputs:
    n           -> Number of Ramps
    T_comb      -> Combustion chamber Temperature
    M0          -> Mach number of free stream flow
    h           -> Flight altitude

    ==> Outputs
    theta       -> Deflection angle of each ramp
    beta_i      -> Wave deflection of each ramp
    P           -> Pressure at each ramp stage
    T           -> Temperature at each ramp stage
    rho         -> Density at each ramp stage
    M           -> Mach number at each ramp stage
    '''
    # Variables initialization (Pressure, Temperature, Density, etc)
    P = np.zeros(n+2)
    T = np.zeros(n+2)
    rho = np.zeros(n+2)
    M = np.zeros(n+2)
    beta_i = np.zeros(n+1)
    theta = np.zeros(n)
    corr = np.zeros(2)

    # Atmospheric properties based on 1976 model
    atmos = atm.ATMOSPHERE_1976(h*1000)

    # Error
    err = 1

    # Initial guess of first ramp angle and its increment
    theta_init = theta_init
    inc = incremento

    print('Beging the optmization process...')
    while abs(err)>erro:
        # Atmospheric pressure [Pa]
        P[0] = atmos.P
        # Local temperature
        T[0] = atmos.T
        # Local density
        rho[0] = atmos.rho
        # Free stream mach
        M[0] = M0
        # First ramp angle (guess)
        theta[0] = theta_init

        # Solving the analytical equations
        for i in range(n+1):

            # Symbolic variables
            beta = Symbol('beta')
            theta_temp = Symbol('theta_temp')

            # Last stage is for reflected show wave (sum of each theta angle)
            if i == n:
                # Beta equation
                eqn = tan(sum(theta)) - ((2/tan(beta))*(M[i]**2*sin(beta)**2 - 1)/(M[i]**2*(k + cos(2*beta))+2))
                beta_i[i] = nsolve(eqn, sum(theta))

                # Mach number after shock wave
                arg = float((1 + (k-1)/2*(M[i]*sin(beta_i[i]))**2)/(k*(M[i]*sin(beta_i[i]))**2 - (k-1)/2))
                M[i+1] = np.sqrt(arg)*(1/sin(beta_i[i] - sum(theta)))

                # Pressure after shock wave
                P[i+1] = P[i]*(1 + (2*k/(k+1))*((M[i]*sin(beta_i[i]))**2-1))

                # Density after shock wave
                rho[i+1] = rho[i]*(M[i]*sin(beta_i[i]))**2*(k+1)/(2+(k-1)*(M[i]*sin(beta_i[i]))**2)

                # Temperature after shock wave
                T[i+1] = T[i]*P[i+1]*rho[i]/(P[i]*rho[i+1])

                break

            # Solving for the first ramp based on theta's initial guess and its updates
            if i == 0:
                # Beta equation
                eqn = tan(theta[i]) - ((2/tan(beta))*(M[i]**2*sin(beta)**2 - 1)/(M[i]**2*(k + cos(2*beta))+2))
                beta_i[i] = nsolve(eqn, theta[i])

                # Mach number after shock wave
                arg = float((1 + (k-1)/2*(M[i]*sin(beta_i[i]))**2)/(k*(M[i]*sin(beta_i[i]))**2 - (k-1)/2))
                M[i+1] = np.sqrt(arg)*(1/sin(beta_i[i] - theta[i]))

                # Pressure after shock wave
                P[i+1] = P[i]*(1 + (2*k/(k+1))*((M[i]*sin(beta_i[i]))**2-1))

                # Density after shock wave
                rho[i+1] = rho[i]*(M[i]*sin(beta_i[i]))**2*(k+1)/(2+(k-1)*(M[i]*sin(beta_i[i]))**2)

                # Temperature after shock wave
                T[i+1] = T[i]*P[i+1]*rho[i]/(P[i]*rho[i+1])

            # Solving from the calculation of the first ramp, the properties and calculating the other thetas
            else:
                beta_i[i] = asin(M[i-1]*sin(beta_i[i-1])/M[i])

                # Theta equation
                eqn = tan(theta_temp) - ((2/tan(beta_i[i]))*(M[i]**2*sin(beta_i[i])**2 - 1)/(M[i]**2*(k + cos(2*beta_i[i]))+2))
                theta[i] = nsolve(eqn, beta_i[i])

                # Mach number after oblique shock wave
                arg = float((1 + (k-1)/2*(M[i]*sin(beta_i[i]))**2)/(k*(M[i]*sin(beta_i[i]))**2 - (k-1)/2))
                M[i+1] = np.sqrt(arg)*(1/sin(beta_i[i] - theta[i]))

                # Pressure after shock wave
                P[i+1] = P[i]*(1 + (2*k/(k+1))*((M[i]*sin(beta_i[i]))**2-1))

                # Density after shock wave
                rho[i+1] = rho[i]*(M[i]*sin(beta_i[i]))**2*(k+1)/(2+(k-1)*(M[i]*sin(beta_i[i]))**2)

                # Temperature after shock wave
                T[i+1] = T[i]*P[i+1]*rho[i]/(P[i]*rho[i+1])

        # Error analysis
        err = (T[-1] - T_comb)/T_comb

        if corr[0] == 0 and corr[1] == 0:
            corr[0] = err
            corr[1] = err

        if err<0:
            corr[0] = err
            err = 1
            if corr[0]*corr[1]<0:
                inc = np.sqrt(inc**2)/2
            theta_init = theta_init + inc

        elif err>erro:
            corr[1] = err
            if corr[0]*corr[1]<0:
                inc = inc/2
            theta_init = theta_init - inc

    theta = np.append(theta, sum(theta))
    return theta, beta_i, P, T, rho, M

def heat_add(P, T, rho, M, M_exit):
    '''
    Code to calculate the operating conditions after heat addition
    P       -> Pressure at the combustion chamber inlet
    T       -> Temperature at the combustion chamber inlet
    rho     -> Specific mass at combustion chamber inlet
    M       -> Mach at the combustion chamber inlet
    '''
    if M_exit < 1.1:
        M_exit = 1.1

    kr = korkegi(M)

    P_exit =P* ((1+k*M**2)/(1+k*M_exit**2)) # P_exit/P
    T_total_inlet = T + 0.5*k*T*R/cp
    if P_exit/P>kr:
        P_exit = kr*P
        M_exit = np.sqrt((P/P_exit * (1+k*M**2) - 1)/k)

    rho_exit = rho*((1+k*M_exit**2)/(1+k*M**2))*(M/M_exit)**2
    T_exit = T*((1+k*M**2)/(1+k*M_exit**2))**2*(M_exit/M)**2

    T_total_exit = T_total_inlet*(((1+k*M**2)/(1+k*M_exit**2))**2*(M_exit/M)**2)*((1+0.5*(k-1)*M_exit**2)/(1+0.5*(k-1)*M**2))

    q = cp*(T_total_exit - T_total_inlet)
    return P_exit, T_exit, rho_exit, M_exit, T_total_exit, T_total_inlet, q

def korkegi(M):
    """
    Korkegi limit

    ==> Inputs:
    M0          -> Mach number

    ==> Inputs:
    Critical pressure ratio
    """

    if M<= 4.5:
        return 1 + 0.3*M**2
    else:
        return 0.17*M**2.5

def print_results(theta, beta, P, T, rho, M):
    print('theta: ', theta*180/np.pi, '[DEG]')
    print('beta: ', beta*180/np.pi, '[DEG]')
    print('P: ', P, '[Pa]')
    print('T: ', T, '[K]')
    print('rho: ', rho, '[kg/m3]')
    print('Ma: ', M)

if __name__ == '__main__':
    # Inputs - These inputs with subindex 0 indicate the input properties
    # And free-flow, as well as the first engine ramp information
    # Hydrogen and air constants
    h_pr = 119.954E6                # Heat of combustion j/kg
    f_st = 0.0291                   # Fuel/air ratio stoichiometric
    cp = 1006.15                    # Specific Heat for air [j/kgK]
    k = 1.4                         # Specific heat ratio of air
    R = 8.3144621                   # Gas constant

    # Number of ramps
    n = 2

    # Desired Temperature at Combustion Chamber [K]
    T_comb = 1000

    # Mach of vehicle
    M0 = 9

    # Desired Mach number at the entrance of the nozzle
    M_ex_comb = 1.2

    # Operating altitude of the vehicle [km]
    h = 35

    # Desired error
    erro = 1E-5

    # Increment of each ramp at each iteration
    incremento = 1.2*np.pi/180

    # Initial guess for the first ramp
    theta_init = 5*np.pi/180

    # Optimizing ramps angles based on previous inputs
    theta, beta, P, T, rho, M = optimize(n, T_comb, M0, h, erro, incremento, theta_init)
    print_results(theta, beta, P, T, rho, M)



    # # Heat addition =================================================================
    #
    # # Resultado obtido do CFD
    # m_dot = 3.967776
    #
    # P_exit, T_exit, rho_exit, M_exit, T_total_exit, T_total_inlet, q = heat_add(P[-1], T[-1], rho[-1], M[-1], M_ex_comb)
    # q_dot = q*m_dot
    # volume = (12/1000)*(1/1000)*1;
    # print("====== Properties after heat add =======")
    # print('Mach exit: ', M_exit)
    # print('P exit: ', P_exit)
    # print('T exit: ', T_exit)
    # print('rho exit: ', rho_exit)
    #
    # print("====== Heat addition ==========")
    # print('q: ', q/1000, '[kJ/kg]')
    # print('q_dot: ', q_dot/1000, '[kW]')
    # print('q_dot_vol:', q_dot/volume, '[W/m^3]')
    # print('Tt_exit: ',T_total_exit)
    # print('Tt_inlet: ',T_total_inlet)
    #
    # q_dot_vol = 640000000000
    # q_dot = 640000000000*volume
    # q = q_dot/m_dot
    # T_total_exit = q/cp + T_total_inlet
    # print('------------------------------')
    # print('T_total necessária: ', T_total_exit)
    # print('q: ', q/1000)
