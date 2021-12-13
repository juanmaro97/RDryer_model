import matplotlib.pyplot as plt
import math
from sklearn import metrics
from scipy.integrate import odeint
import numpy as np

""" ------ SIMULACIÓN DE SECADOR ROTATORIO ------ """
### Abbasfard H. (et. al) (2013), 
# Mathematical modeling and simulation of an industrial rotary dryer: A case study of
# ammonium nitrate plant

def humedad_solidos(Xs,z,k,Xe,M,S):
    """this is the rhs of the ODE to integrate, i.e. dV/dt=f(V,t)"""
    '''
    Xs: Solid moisture []
    z: Dimensionless position []
    k:  Drying constant
    Xe: Equilibrium moisture []
    M: Total load [kg]
    S: Solid flow in [kg/s]
    '''

    return -((k*(Xs-Xe))*M)/S


def humedad_aire(Y,z,k,Xs,Xe,M,G):
    """this is the rhs of the ODE to integrate, i.e. dV/dt=f(V,t)"""
    '''
    Y: Solid moisture []
    Xs: Solid moisture []
    z: Dimensionless position []
    k:  Drying constant
    Xe: Equilibrium moisture []
    M: Total load [kg]
    G: Air flow in [kg/s]
    '''

    return ((k*(Xs-Xe))*M)/G


def temp_solido(Ts,z,Q,lamda,R,M,S,Cpd,Xs,Cpw):
    """this is the rhs of the ODE to integrate, i.e. dV/dt=f(V,t)"""
    '''
    Ts: Solid temperature [°C]
    Xs: Solid moisture []
    Q: Heat transfer
    lamda: Latent heat of vaporization [kJ/kg]
    R: Drying rate [kg water/kg dry solid/s]
    M: Total load [kg]
    S: Solid flow in [kg/s]
    Cpd: Specific heat of dry solid [kJ/kg °C]
    Cpw: Specific heat of pure water [kJ/kg °C]
    '''

    return (-Q-lamda*R*M)/(S*(Cpd+Xs*Cpw))


def temp_aire(Tg,z,Q,Cpv,R,M,Ts,Qp,G,Cpg,Y):
    """this is the rhs of the ODE to integrate, i.e. dV/dt=f(V,t)"""
    '''
    Tg: Air temperature [°C]
    Y: Air moisture []
    Q: Heat transfer
    lamda: Latent heat of vaporization [kJ/kg]
    R: Drying rate [kg water/kg dry solid/s]
    M: Total load [kg]
    G: Air flow in [kg/s]
    Cpv: Specific heat of water vapor [kJ/kg °C]
    Cpg: Specific heat of dry air [kJ/kg °C]
    '''

    return (Q+(Cpv*R*M*(Ts-Tg))-Qp)/(G*(Cpg+Y*Cpv))


def heat_transfer(Uva,V,Ts,Tg,Tamb,Up,A):

    Q = Uva*V*(Ts-Tg)
    Qp = Up*A*(Tg-Tamb)

    return Q, Qp

def volumetric_heat_transfer_coeff(G,A,S):
    Uva = (0.394*(G/A)**0.289)*((S/A)**0.541)
    # Uva = (3.535*(G/A)**0.289)*((S/A)**0.541)
    Up = 0.022*(G/A)**0.879
    # Up = 0.227*(G/A)**0.879

    return Uva,Up

def eq_moisture(RH,Tg):
    a = (2.39*10**(-6))*(0.987**Tg)*Tg**(-0.832)
    b = (-5.76*10**(-5))+(1.306*10**(-5))*np.log(Tg)
    c = 0.9715*(1.024**Tg)*Tg**(-2.31)

    Xeq = RH*(a*RH*RH+b*RH+c)

    return Xeq

def drying_constant(Tg):

    k = 0.0349*math.exp(-7.95/Tg)

    return k

def residence_time(L,slope,n,S,D,Dp,G):

    # tau = (13.8*L)/(slope*(n**0.9)*D)-((614.2)/(Dp**0.5))*((L*G)/S)
    tau = (0.35*L)/(slope*(n**0.9)*D)-((5.2)/(Dp**0.5))*((L*G)/S)

    return tau


Tamb = 35

S = 60001/3600          # Solid flow in [kg/s]
G = 60979/3600          # Air flow in [kg/s]

L = 18                  # Dryer lenght [m]
D = 3.324               # Dryer inside diameter [m]
slope = 0.025           # Dryer slope [m/m]
n = 3                   # RPM
Dp = 2*10**3            # Average solid diameter [um]
A = np.pi*(D**2)/4      # Transversal area [m2]
V = A*L                 # Dryer volume [m3]


Cpd = 1.56              # Solid heat capacity [kJ/kg°C]
Cpg = 1.009             # Air heat capacity [kJ/kg°C]
Cpv = 1.88              # Vapor heat capacity [kJ/kg°C]
Cpw = 4.18              # Water heat capacity [kJ/kg°C]

lamda = 2260            # Latent heat of vaporization [kJ/kg]

Xs0 = 0.12              # Input solid moisture []
Y0 = 0.005              # Input air moisture []
Ts0 = 84                # Input solid temperature [°C]
Tg0 = 73                # Input air temperature [°C]
RH = 0.015              # Input air relative humidity []



Uva, Up = volumetric_heat_transfer_coeff(G,A,S)
Q, Qp = heat_transfer(Uva,V,Ts0,Tg0,Tamb,Up,A)
k = drying_constant(Tg0)
Xe = eq_moisture(RH,Tg0)
tau = residence_time(L,slope,n,S,D,Dp,G)

M = S*tau

R = k*(Xs0-Xe)


args_Xs = (k,Xe,M,S)
args_Y = (k,Xs0,Xe,M,G)
args_Ts = (Q,lamda,R,M,S,Cpd,Xs0,Cpw)
args_Tg = (Q,Cpv,R,M,Ts0,Qp,G,Cpg,Y0)

sample_time = 0.01
z = np.arange(0, 1, sample_time)  # values of t for
                        # which we require
                        # the solution y(t)

Xs_temp = np.zeros(z.shape)
Y_temp = np.zeros(z.shape)
Ts_temp = np.zeros(z.shape)
Tg_temp = np.zeros(z.shape)
print("K es: ",k)
print("Xe inciial es: ",Xe)
print("Tau inciial es: ",tau)
print("Q inciial es: ",Q)
print("Uva inciial es: ",Uva)
print("Volumen secador es: ",V)

"""" ----- SIMULACIÓN ------ """

for k in range(z.size):
    # print("\n ------------------CICLO ",k)
    # print("\n k es: ",k)

    Xs_temp[k] = Xs0
    Ts_temp[k] = Ts0
    Y_temp[k] = Y0
    Tg_temp[k] = Tg0


    # print("first X0: " ,Xs0)
    # print("this is first args_Xs, ",args_Xs)
    Xs = odeint(humedad_solidos, Xs0, np.array([0.0, sample_time]), args=args_Xs)
    # print("this is Xs: ",Xs)
    

    # print("\n first Y0: " ,Y0)
    # print("this is first args_Y, ",args_Y)
    Y = odeint(humedad_aire, Y0, np.array([0.0, sample_time]), args=args_Y)
    # print("this is Y: ",Y[1])
    
    
    # print("\n first Ts: " ,Ts0)
    # print("this is first args_Ts, ",args_Ts)
    Ts = odeint(temp_solido, Ts0, np.array([0.0, sample_time]), args=args_Ts)
    # print("this is Ts: ",Ts[1])
    
    
    # print("\n first Ta: " ,Tg0)
    # print("this is first args_Ta, ",args_Tg)
    Tg = odeint(temp_aire, Tg0, np.array([0.0, sample_time]), args=args_Tg)
    # print("this is Ta: ",Tg[1])
    
    Xs0 = Xs[1]
    Y0 = Y[1]
    Ts0 = Ts[1]
    Tg0 = Tg[1]

    Q, Qp = heat_transfer(Uva,V,Ts0,Tg0,Tamb,Up,A)
    k = drying_constant(Tg0)
    Xe = eq_moisture(RH,Tg0)

    R = k*(Xs0-Xe)

    args_Xs = (k,Xe,M,S)
    args_Y = (k,Xs0,Xe,M,G)
    args_Ts = (Q,lamda,R,M,S,Cpd,Xs0,Cpw)
    args_Tg = (Q,Cpv,R,M,Ts0,Qp,G,Cpg,Y0)
    

    # print("\n this is Xe: ",Xe)
    # print("this is K: ",k)
    # print("this is Q: ",Q)
    # print("this is Qp: ",Qp)

delta_T = np.abs(Ts0-Tg0)
    

"""" ------ PLOTTING ------- """

fig, axarr = plt.subplots(2, 1)
fig.canvas.set_window_title('Rotary dryer') 
fig.suptitle('Rotary dryer simulation')

axarr[0].plot(z, 100*Xs_temp, 'b', label='Solid')
axarr[0].plot(z, 100*Y_temp, 'green', label='Air')
axarr[0].legend(loc='best')
axarr[0].grid()
axarr[0].set(ylabel = 'Moisture [%]')

axarr[1].plot(z, Ts_temp, 'r', label='Solid')
axarr[1].plot(z, Tg_temp, 'y', label='Air')
axarr[1].legend(loc='best')
axarr[1].grid()
axarr[1].set(ylabel = 'Temperature [°C]')
axarr[1].set_title("Delta T is: " + str(delta_T))

plt.show()