import random
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.random import rand
from scipy.ndimage.filters import gaussian_filter1d


class Ising2D:
    r"""
    Class for the 2D Ising model.

    Parameters
    ----------
    L : int
        length of the square lattice.
    T : float
        temperature.
    nsteps : int
        number of Monte Carlo steps.
    J : float
        exchange constant.
    H : array-like
        magnetic field.
    """
    def __init__(self, L, T, nsteps, J, H):
        r"""
        Initializes the Heisenberg2D object with the specified parameters.
        """


        self.L = L 
        self.T = T 
        self.nsteps = nsteps 
        self.J = J 
        self.H = H 
        self.N = L * L
        self.beta = 1.0 / T
        self.nbr = {i: ((i // L) * L + (i + 1) % L, (i + L) % self.N,
                        (i // L) * L + (i - 1) % L, (i - L) % self.N) \
                    for i in range(self.N)} # vecinos de cada sitio
        self.S = [random.choice([1, -1]) for k in range(self.N)] # #Escribe una cadena de N elementos y a cada sitio le asigna un valor aleatorio de +1 o -1
        self.R = [random.choice([1, -1]) for k in range(self.N)]
        self.iso = self.ising_2d()

    def iso2(self):
        r"""
        Performs an inversion on the magnetic field H and returns a new configuration.

        Returns:
        - iso2 : tuple
            results of the Ising model with inverted magnetic field.
        """

        self.H = np.flip(self.H)
        iso2 = self.ising_2d()
        return iso2

    def x_y(self, k):
        r"""
        Converts an index k to the corresponding (x, y) coordinates in the square lattice.

        Parameters
        ----------
        k : int
            index of the site in the lattice.

        Returns:
        x : int
            x-coordinate corresponding to index k.
        y: int
            y-coordinate corresponding to index k.
        """
        y = k // self.L
        x = k - y * self.L
        return x, y

    # funcion para el modelo de Ising
    def ising_2d(self):
        r"""
        Implements the 2D Ising model.

        Returns
        -------
        iso : tuple
            results of the Ising model.
        R : array
            initial configuration of random spins.
        conf1 : array
                initial configuration of spins R.
        S : array
            final configuration of spins after nsteps of Monte Carlo.
        conf : array
            final configuration of spins S.
        avg_energy : float
            average energy per site.
        E : list
            list of energy per site at each Monte Carlo step.
        Mag : list
            list of magnetizations per site at each Monte Carlo step.
        E_1 : list
            cumulative energy at each Monte Carlo step.
        itera : list
            list of Monte Carlo steps.
        conf_int : array
            intermediate configuration of spins.
        """
        energy = 0
        N = self.L * self.L
        beta = 1.0 / self.T
        nbr = {i: ((i // self.L) * self.L + (i + 1) % self.L, (i + self.L) % N,
                   (i // self.L) * self.L + (i - 1) % self.L, (i - self.L) % N) \
               for i in range(N)}
        S = self.S.copy()
        R = self.R.copy()

        for k in range(N):
            energy += S[k] * sum(S[nn] for nn in nbr[k])
        energy *= 0.5

        E = []
        E_1 = []
        Mag = []
        itera = []

        for i in tqdm(range(len(self.H))):
            #Guardar una copia de los snapshots en una parte intermedia para visualizar las configuraciones
            if i==int(len(self.H)*0.475):
                Intermedio=S.copy()
            for step in range(self.nsteps):
                k = random.randint(0, N - 1)
                delta_E = 2.0 * self.J * S[k] * sum(S[nn] for nn in nbr[k]) + self.H[i] * S[k]
                if random.uniform(0.0, 1.0) < math.exp(-beta * delta_E):
                    S[k] *= -1
                    energy += delta_E
                else:
                    S[k] = S[k]
                E_1.append(energy)
                itera.append(step)
            E.append(energy)

            M = np.sum(S) / N
            Mag.append(M)
        conf=[[0 for x in range (self.L)] for y in range (self.L)]#Configuración inicial
        conf1=[[0 for x in range (self.L)] for y in range (self.L)]#Configuración inicial
        conf_int=[[0 for x in range (self.L)] for y in range (self.L)]#Configuración intermedia
        for k in range(N):
            x, y = self.x_y(k)
            conf[x][y] = S[k]
        for k in range(N):
            X, Y = self.x_y(k)
            conf1[X][Y] = R[k]
        for k in range (N):
            x1,y1=self.x_y(k)
            conf_int[x1][y1]=Intermedio[k]

        return R, conf1, S, conf, sum(E) / float(len(E) * N), E, Mag, E_1, itera, conf_int


    # función de ajuste al modelo de tangente hiperbólica

    def tanh(self, x, a, b, c):
        r"""
        Hyperbolic tangent function used for curve fitting.
        """
        return a * np.tanh(b * x + c)

    # Ajustemos la curva de la isotermas para T1 T2 y T3 y grafiquemos con subplots
    def plot_ajuste(self):
        """
        Fits the isotropic curves to a hyperbolic tangent model and plots the results.
        """
        popt, pcov = curve_fit(self.tanh, self.H, self.iso[6])

        # graficar ajuste
        plt.figure(figsize=(10, 8))
        plt.title("Isotermas  y ajuste para un ferromagneto en el modelo de ising en 2D", fontsize=12)
        plt.plot(self.H, self.iso[6], label="T={}K".format(self.T), color="red")
        plt.plot(self.H, self.tanh(self.H, *popt), '*', label="Ajuste", color="black")
        plt.grid()
        plt.legend()
        plt.show()

        return

    # graficar histeresis
    def plot_histeresis(self):
        """
        Plots the hysteresis loop for the Ising model.
        """
        plt.figure(figsize=(10, 6))
        plt.title("Histeresis para un ferromagneto en el modelo de ising en 2D", fontsize=20)
        plt.plot(np.arange(self.H[0], self.H[-1], 0.5), self.iso[6], label="T={}K".format(self.T), color="green")
        plt.plot(np.arange(self.H[-1], self.H[0], -0.5), self.iso2()[6], label="T={}K".format(self.T), color="black")
        plt.xlabel("H", fontsize=20)
        plt.ylabel("M", fontsize=20)
        plt.xlim(-5, 5)
        plt.grid()
        plt.legend()
        plt.show()

        return

    # graficar magnetizacion
    def plot_magnetizacion(self):
        r"""
        Plots the magnetization as a function of the magnetic field.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.H, self.iso[6], 'o', color='red')
        plt.xlabel('Campo magnético')
        plt.ylabel('Magnetización')
        plt.title('Magnetización vs Campo magnético')
        plt.show()

        return

    # graficar energía
    def plot_energia(self):
        """
        Plots the energy as a function of the magnetic field.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.H, self.iso[5], 'o', color='blue')
        plt.xlabel('Campo magnético')
        plt.ylabel('Energía')
        plt.title('Energía vs Campo magnético')
        plt.show()

        return

    # graficar energía en función de las iteraciones
    def plot_energia_itera(self):
        """
        Plots the energy as a function of the Monte Carlo iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.iso[8], self.iso[7], 'o', color='blue')
        plt.xlabel('Iteración')
        plt.ylabel('Energía')
        plt.title('Energía vs Iteración')
        plt.show()

        return

    def plot_isotermas(self):
        plt.figure(figsize=(10, 6))
        plt.title("Isotermas para un ferromagneto en el modelo de ising en 2D", fontsize=14)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.plot(self.H, self.iso[6], label="T={}K".format(self.T), color="red")
        #plt.plot(self.H, Iso2_1[6], label="T={}K".format(T2), color="blue")
        #plt.plot(self.H, Iso3_1[6], label="T={}K".format(T3), color="green")
        plt.xlabel("H", fontsize=20)
        plt.ylabel("M", fontsize=20)
        plt.legend(fontsize=15, loc="best")
        plt.grid()
        plt.show()

    # graficar configuraciones
    def plot_configuraciones(self):
        """
        Plots the initial, intermediate, and final configurations of spins.
        """
        plt.figure(figsize=(12, 6))
        plt.title('Snapshots de las Configuraciones')
        plt.subplot(1,3,1)
        plt.imshow(self.iso[1], extent=[0, self.L, 0, self.L], interpolation='nearest')
        plt.xlabel("x",fontsize=10)
        plt.ylabel("y",fontsize=10)
        plt.title("Configuración inicial",fontsize=10)
        plt.subplot(1,3,2)
        plt.imshow(self.iso[9], extent=[0, self.L, 0, self.L], interpolation='nearest')
        plt.xlabel("x",fontsize=10)
        plt.ylabel("y",fontsize=10)
        plt.title("Configuración intermedia",fontsize=10)
        plt.subplot(1,3,3)
        plt.imshow(self.iso[3], extent=[0, self.L, 0, self.L], interpolation='nearest')
        plt.xlabel("x",fontsize=10)
        plt.ylabel("y",fontsize=10)
        plt.title("Configuración final",fontsize=10)
        plt.show()

        return

# clase heredada para realizar las gráficas de Energía, Magnetización y Calor especifíco en función de la temperatua
class IsingTemp(Ising2D):
    def __init__(self, L, T, nsteps, J, H, nt, eq_steps, mc_steps, t_array):
        """hola"""
        super().__init__(L, T, nsteps, J, H)
        self.nt = nt
        self.eq_steps = eq_steps
        self.mc_steps = mc_steps
        self.N = int(np.sqrt(L))
        self.M = np.zeros(self.nt)
        self.C = np.zeros(self.nt)
        self.X = np.zeros(self.nt)
        self.E = np.zeros(self.nt)
        self.n1 = 1.0 / (mc_steps * self.N * self.N)
        self.n2 = 1.0 / (mc_steps * mc_steps * self.N * self.N)
        self.t_array = t_array


    def initial_state(self):
        """rGenera un estado aleatorio para una red de NxN """

        state = 2 * np.random.randint(2, size=(int(self.N), int(self.N))) - 1
        return state

    def mc_move(self, config, beta):
        r"""Movimiento de Monte Carlo usando el algoritmo de Metropolis"""

        for i in range(self.N):
            for j in range(self.N):
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                s = config[a, b]
                nb = config[(a + 1) % self.N, b] + config[a, (b + 1) % self.N] + config[(a - 1) % self.N, b] + config[a, (b - 1) % self.N]
                cost = 2 * s * nb

                if cost < 0:
                    s *= -1
                elif np.random.rand() < np.exp(-cost * beta):
                    s *= -1
                config[a, b] = s
        return config

    def calc_energy(self, config):
        r"""Energía de una configuración dada"""

        energy = 0

        for i in range(len(config)):
            for j in range(len(config)):
                S = config[i, j]
                nb = config[(i+1)%self.N, j] + config[i,(j+1)%self.N] + config[(i-1)%self.N, j] + config[i,(j-1)%self.N]
                energy += -nb * S

        return energy/2.

    def calc_mag(self,config):
        r"""Magnetización de una configuración  dada"""

        mag = np.sum(config)
        return mag

    def ciclo_t(self):
        for tt in tqdm(range(self.nt)):
            config = self.initial_state()

            E1 = M1 = E2 = M2 = 0
            iT = 1.0 / self.t_array[tt]
            iT2 = iT * iT

            for i in range(self.eq_steps):
                self.mc_move(config, iT)

            for i in range(self.mc_steps):
                self.mc_move(config, iT)
                ene = self.calc_energy(config)
                mag = self.calc_mag(config)

                E1 = E1 + ene
                M1 = M1 + mag
                M2 = M2 + mag * mag
                E2 = E2 + ene * ene

        # Guardar los valores de energía y magnetización
            self.E[tt] = self.n1 * E1
            self.M[tt] = self.n1 * M1
            self.C[tt] = (self.n1 * E2 - self.n2 * E1 * E1) * iT2
            self.X[tt] = (self.n1 * M2 - self.n2 * M1 * M1) * iT

        plt.figure(figsize=(10, 6))
        plt.title('Energía vs Temperatura', fontsize = 15)
        plt.scatter(self.t_array,  self.E, s = 50, marker='o', label = 'Energía', color = 'IndianRed')
        plt.plot(self.t_array, gaussian_filter1d(self.E, sigma = 1), label = 'Ajuste', color = 'darkblue')
        plt.xlabel('Temperatura (K)', fontsize = 15)
        plt.ylabel('Energía', fontsize = 15)
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.title('Magnetización vs Temperatura', fontsize=15)
        plt.scatter(self.t_array, abs(self.M), s=50, marker='o', label='Magnetización', color='IndianRed')
        plt.plot(self.t_array, gaussian_filter1d(abs(self.M), sigma=1), label='Ajuste', color='darkblue')
        plt.xlabel('Temperatura (K)', fontsize=15)
        plt.ylabel('Magnetización', fontsize=15)
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.title('Capacidad Calorífica vs Temperatura', fontsize=15)
        plt.scatter(self.t_array, self.C, s=50, marker='o', label='Capacidad Calorífica', color='IndianRed')
        plt.plot(self.t_array, gaussian_filter1d(self.C, sigma=1), label='Ajuste', color='darkblue')
        plt.xlabel('Temperatura (K)', fontsize=15)
        plt.ylabel('Capacidad Calorífica', fontsize=15)
        plt.grid()
        plt.legend()
        plt.show()



        return