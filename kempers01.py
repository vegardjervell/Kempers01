'''
Author: Vegard Gjeldvik Jervell
Date: December 2021
Purpose: Implementation of the model proposed by Kempers (J. Chem. Phys. 115, 6330, 2001) for prediction of the Soret coefficient
         doi : http://dx.doi.org/10.1063/1.1398315
Requires: numpy, scipy, ThermoPack (https://github.com/SINTEF/thermopack), KineticGas (https://github.com/vegardjervell/Kineticgas)
'''
import warnings, platform
import numpy as np
from scipy.constants import gas_constant, Avogadro, Boltzmann
from scipy.optimize import root
from pyctp import cubic
from pykingas import KineticGas


class Kempers(object):
    def __init__(self, comps, eos, alpha_t0_N = 7, sigma=None, eps_div_k=None, mole_weights=None):
        '''
        :param comps (str): comma separated list of components
        :param eos (ThermoPack): Initialized Equation of State object, initialized with components 'comp'
        :param alpha_t0_N (int, optional): Order of approximation of Enskog solutions
        :param sigma (array, optional): Hard sphere diameters [m]
        :param eps_div_k (array, optional): Interaction potential well depth [k_B]
        :param mole_weights (array, optional): Molar masses [g / mol]
        '''

        self.comps = comps
        self.total_moles = 1 #Dummy value, because some ThermoPack methods require it. Does not effect output.
        eoscomp_inds = [eos.getcompindex(comp) for comp in self.comps.split(',')]
        if -1 in eoscomp_inds:
            warnings.warn('Equation of state and model must be initialized with same components.\n'
                          "I'm initializing using SRK with "+comps+" now to avoid crashing")
            self.eos = cubic.cubic()
            self.eos.init(self.comps, 'SRK')

        elif any(np.array(eoscomp_inds) != sorted(np.array(eoscomp_inds))):
            eoscomps = ','.join(eos.get_comp_name(i) for i in sorted(eoscomp_inds))
            warnings.warn('Equation of state and model must be initialized with same components in the same order\n'
                            'but are initialized with ' + eoscomps + ' and ' + self.comps+'.\n'
                            "I'm initializing using SRK with "+comps+" now to avoid crashing")
            self.eos = cubic.cubic()
            self.eos.init(self.comps, 'SRK')

        else:
            self.eos = eos

        if mole_weights is None:
            self.mole_weights = np.array([self.eos.compmoleweight(self.eos.getcompindex(comp)) for comp in comps.split(',')])
        else:
            self.mole_weights = np.copy(mole_weights)

        self.kinetic_gas = KineticGas(comps, mole_weights=self.mole_weights, sigma=sigma, eps_div_k=eps_div_k)
        self.alpha_t0_N = alpha_t0_N

    def get_binary_soret_cov(self, T, p, x, phase, kin=False, BH=False):
        '''
            Get soret coefficients at specified phase point, for binary mixture, center of volume frame of reference
            :param T (float): Temperature [K]
            :param p (float): Pressure [Pa]
            :param x (ndarray): Composition [mole fraction]
            :param phase (int): ThermoPack phase key
            :param kin (bool, optional): Return kinetic gas value?
            :param BH (bool, optional): Use Barker-Henderson diameters?
            :return: (ndarray) soret coefficients
        '''

        R = gas_constant
        v, dvdn = self.eos.specific_volume(T, p, x, phase, dvdn=True)
        h, dhdn = self.eos.enthalpy(T, p, x, phase, dhdn=True)
        h0, dh0dn = self.eos.enthalpy(T, 1e-5, x, 2, dhdn=True)

        dmudx = self.dmudx_TP(T, p, x, phase)
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH, N=self.alpha_t0_N)

        v1, v2 = dvdn
        h1, h2 = dhdn
        h10, h20 = dh0dn

        x1, x2 = x
        dmu1dx1 = dmudx[0, 0]

        alpha_1 = (v1 * v2 / (v1 * x1 + v2 * x2)) * (((h2 - h20) / v2) - ((h1 - h10) / v1)) / (x1 * dmu1dx1) + (R * T * alpha_T0[0]) / (x1 * dmu1dx1)

        alpha = np.array([alpha_1, - alpha_1])

        soret = alpha / T
        if kin is True:
            kin_contrib = alpha_T0 / T  # * R /(self.x * dmudx.diagonal())
            return soret, kin_contrib
        else:
            return soret

    def get_soret_cov(self, T, p, x, phase, kin=False, BH=False):
        '''
        Get soret coefficients at specified phase point, center of volume frame of reference
            :param T (float): Temperature [K]
            :param p (float): Pressure [Pa]
            :param x (ndarray): Composition [mole fraction]
            :param phase (int): ThermoPack phase key
            :param kin (bool, optional): Return kinetic gas value?
            :param BH (bool, optional): Use Barker-Henderson diameters?
        :return: (ndarray) soret coefficients
        '''

        if len(x) == 2:
            return self.get_binary_soret_cov(T, p, x, phase, kin=kin)

        R = gas_constant
        v, dvdn = self.eos.specific_volume(T, p, x, phase, dvdn=True)
        h, dhdn = self.eos.enthalpy(T, p, x, phase, dhdn=True)
        h0, dh0dn = self.eos.enthalpy(T, 1e-5, x, 2, dhdn=True)

        dmudx = self.dmudx_TP(T, p, x, phase)
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH, N=self.alpha_t0_N)

        #using alpha_T0 as initial guess for root solver
        initial_guess = alpha_T0

        N = len(self.x)
        #Defining the set of equations
        def eq_set(alpha):
            eqs = np.zeros(N)
            for i in range(N-1):
                eqs[i] = ((dhdn[-1] - dh0dn[-1])/dvdn[-1]) - ((dhdn[i] - dh0dn[i])/dvdn[i])\
                         + R * T * ((alpha_T0[i] * (1 - x[i]) / dvdn[i])
                         - (alpha_T0[-1] * (1 - x[-1])/dvdn[-1]))\
                         - sum((dmudx[i, j]/dvdn[i] - dmudx[-1, j]/dvdn[-1]) * x[j] * (1 - x[j]) * alpha[j]
                               for j in range(N - 1))

            eqs[N-1] = sum(x * (1 - x) * alpha)
            return eqs

        #Solve the set of equations, warn if non-convergent
        solved = root(eq_set, initial_guess)
        if solved.success is False:
            warnings.warn('Solution did not converge for composition :' + str(x) + ', Temperature :'+ str(T) + ', Pressure : '+str(p/1e5))
            alpha = np.full_like(solved.x, np.nan)
            alpha = solved.x
        else:
            alpha = solved.x

        soret = alpha / T
        if kin is True:
            kin_contrib = alpha_T0 / T #* R /(self.x * dmudx.diagonal())
            return soret, kin_contrib
        else:
            return soret

    def get_binary_soret_com(self, T, p, x, phase, kin=False, BH=False):
        '''
        Get soret coefficients at specified phase point, for binary mixture, center of mass frame of reference
            :param T (float): Temperature [K]
            :param p (float): Pressure [Pa]
            :param x (ndarray): Composition [mole fraction]
            :param phase (int): ThermoPack phase key
            :param kin (bool, optional): Return kinetic gas value?
            :param BH (bool, optional): Use Barker-Henderson diameters?
        :return: (ndarray) soret coefficients
        '''

        R = gas_constant
        v, = self.eos.specific_volume(T, p, x, phase)
        h, dhdn = self.eos.enthalpy(T, p, x, phase, dhdn=True)
        h0, dh0dn = self.eos.enthalpy(T, 1e-5, x, 2, dhdn=True)

        dmudx = self.dmudx_TP(T, p, x, phase)
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH, N=self.alpha_t0_N)

        m1, m2 = self.mole_weights
        h1, h2 = dhdn
        h10, h20 = dh0dn
        x1, x2 = x
        dmu1dx1 = dmudx[0, 0]

        alpha_1 = (m1 * m2 / (m1 * x1 + m2 * x2)) * (((h2 - h20) / m2) - ((h1 - h10) / m1)) / (x1 * dmu1dx1) + (
                    R * T * alpha_T0[0]) / (x1 * dmu1dx1)

        alpha = np.array([alpha_1, - alpha_1])

        soret = alpha / T
        if kin is True:
            kin_contrib = alpha_T0 / T  # * R /(self.x * dmudx.diagonal())
            return soret, kin_contrib
        else:
            return soret

    def get_soret_com(self, T, p, x, phase, kin=False, BH=False):
        '''
        Get soret coefficients at specified phase point, center of mass frame of reference
            :param T (float): Temperature [K]
            :param p (float): Pressure [Pa]
            :param x (ndarray): Composition [mole fraction]
            :param phase (int): ThermoPack phase key
            :param kin (bool, optional): Return kinetic gas value?
            :param BH (bool, optional): Use Barker-Henderson diameters?
        :return: (ndarray) soret coefficients
        '''

        if len(x) == 2:
            return self.get_binary_soret_com(T, p, x, phase, kin=kin, BH=BH)

        R = gas_constant
        M = self.mole_weights
        v, = self.eos.specific_volume(T, p, x, phase)
        h, dhdn = self.eos.enthalpy(T, p, x, phase, dhdn=True)
        h0, dh0dn = self.eos.enthalpy(T, 1e-5, x, 2, dhdn=True)

        dmudx = self.dmudx_TP(T, p, x, phase)
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH, N=self.alpha_t0_N)

        # using alpha_T0 as initial guess for root solver
        initial_guess = alpha_T0

        N = len(self.x)

        # Defining the set of equations
        def eq_set(alpha):
            eqs = np.zeros(N)
            for i in range(N - 1):
                eqs[i] = ((dhdn[-1] - dh0dn[-1]) / M[-1]) - ((dhdn[i] - dh0dn[i]) / M[i]) \
                            + R * T * ((alpha_T0[i] * (1 - x[i]) / M[i])
                            - (alpha_T0[-1] * (1 - x[-1]) / M[-1])) \
                            - sum((dmudx[i, j] / M[i] - dmudx[-1, j] / M[-1]) * x[j] * (1 - x[j]) * alpha[j]
                    for j in range(N - 1))

            eqs[N - 1] = sum(x * (1 - x) * alpha)
            return eqs

        # Solve the set of equations, warn if non-convergent
        solved = root(eq_set, initial_guess)
        if solved.success is False:
            warnings.warn('Solution did not converge for composition :'+str(x)+ ', Temperature :' + str(T))
            alpha = np.full_like(solved.x, np.nan)
        else:
            alpha = solved.x

        soret = alpha / T

        if kin is True:
            kin_contrib = alpha_T0 / T #* R /(self.x * dmudx.diagonal())
            return soret, kin_contrib
        else:
            return soret

    def get_soret(self, T, p, x, phase, mode, kin=False, BH=False):
        '''
        Get soret coefficients at specified phase point
            :param T (float): Temperature [K]
            :param p (float): Pressure [Pa]
            :param x (ndarray): Composition [mole fraction]
            :param phase (int): ThermoPack phase key
            :param mode (str): 'com' or 'cov' to determine mode
            :param kin (bool, optional): Return kinetic gas value?
            :param BH (bool, optional): Use Barker-Henderson diameters?
        :return: (ndarray) Soret coefficients
        '''
        if mode == 'cov':
            return self.get_soret_cov(T, p, x, phase, kin=kin, BH=BH)
        elif mode == 'com':
            return self.get_soret_com(T, p, x, phase, kin=kin, BH=BH)

    def dmudn_TP(self, T, p, x, phase):
        '''
        Calculate chemical potential derivative with respect to number of moles at constant temperature and pressure
        :return: ndarray, dmudn[i,j] = dmu_i / dn_j
        '''

        v, dvdn = self.eos.specific_volume(T, p, x, phase, dvdn=True)
        mu, dmudn_TV = self.eos.chemical_potential_tv(T, v * self.total_moles,
                                                      x * self.total_moles, dmudn=True)
        pres, dpdn = self.eos.pressure_tv(T, v * self.total_moles, x * self.total_moles, dpdn=True)

        return dmudn_TV - np.tensordot(dpdn, dvdn, axes=0)

    def dmudx_TP(self, T, p, x, phase):
        '''
        Calculate chemical potential derivative with respect to mole fraction of components
        at constant temperature and pressure
        :return: ndarray, dmudx[i,j] = dmu_i / dx_j
        '''
        x = np.array(x)
        dmudn = self.dmudn_TP(T, p, x, phase)
        return dmudn * self.total_moles
