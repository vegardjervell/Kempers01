'''
Author: Vegard G. Jervell
Date: December 2020
Purpose: Parent class containing some general procedures to be used in both Kempers89 and Kempers01
Requires: numpy, ThermoPack
Note: This is a virtual class, and will not do anything exciting if initialized on its own.
'''

import numpy as np
from pyctp import cubic
from scipy.constants import gas_constant
import warnings, platform

class Kempers(object):
    def __init__(self, comps, eos, mole_weights=None):
        '''
        model parent class, contains interface for retrieving soret-coefficient for spans of temperatures, pressures or compositions
        and some general initialization procedures that are common for the two model' models
        :param comps (str): comma separated list of components
        :param x (1darray): list of mole fractions
        :param eos (ThermoPack): Initialized Equation of State object, initialized with components 'comp'
        :param temp (float > 0): Temperature [K]
        :param pres (float > 0): Pressure [Pa]
        :param phase: Phase of mixture, used for calculating dmudn_TP, see thermo.thermopack for phase identifiers
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

    def get_binary_soret_cov(self, T, p, x, phase, kin=False, BH=False):
        '''
                Get soret coefficients at current settings, center of volume frame of reference
                :return: (ndarray) soret coefficients
                '''

        R = gas_constant
        v, dvdn = self.eos.specific_volume(T, p, x, phase, dvdn=True)
        h, dhdn = self.eos.enthalpy(T, p, x, phase, dhdn=True)
        h0, dh0dn = self.eos.enthalpy(T, 1e-5, x, 2, dhdn=True)

        dmudx = self.dmudx_TP(T, p, x, phase)
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH)

        v1, v2 = dvdn
        h1, h2 = dhdn
        h10, h20 = dh0dn

        ###################
        # h10 = self.eos.idealenthalpysingle(T, p, 1)
        # h20 = self.eos.idealenthalpysingle(T, p, 2)
        # print(h10, h20)
        ###################

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
        Get soret coefficients, center of volume frame of reference
        :return: (ndarray) soret coefficients
        '''

        if len(x) == 2:
            return self.get_binary_soret_cov(T, p, x, phase, kin=kin)

        R = gas_constant
        v, dvdn = self.eos.specific_volume(T, p, x, phase, dvdn=True)
        h, dhdn = self.eos.enthalpy(T, p, x, phase, dhdn=True)
        h0, dh0dn = self.eos.enthalpy(T, 1e-5, x, 2, dhdn=True)

        dmudx = self.dmudx_TP(T, p, x, phase)
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH)

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
        Get soret coefficients at current settings, center of volume frame of reference
        :return: (ndarray) soret coefficients
        '''

        R = gas_constant
        v, = self.eos.specific_volume(T, p, x, phase)
        h, dhdn = self.eos.enthalpy(T, p, x, phase, dhdn=True)
        h0, dh0dn = self.eos.enthalpy(T, 1e-5, x, 2, dhdn=True)

        dmudx = self.dmudx_TP(T, p, x, phase)
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH)

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
        Get soret coefficients at current settings, center of mass frame of reference
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
        alpha_T0 = self.kinetic_gas.alpha_T0(T, v, x, BH=BH)

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
        if mode == 'cov':
            return self.get_soret_cov(T, p, x, phase, kin=kin, BH=BH)
        elif mode == 'com':
            return self.get_soret_com(T, p, x, phase, kin=kin, BH=BH)

    def dmudn_TP(self, T, p, x, phase):
        '''
        Calculate chemical potential derivative with respect to number of moles at constant temperature and pressure
        :return: ndarray, dmudn[i,j] = dmu_idn_j
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
        :return: ndarray, dmudx[i,j] = dmu_idn_j
        '''
        x = np.array(x)
        dmudn = self.dmudn_TP(T, p, x, phase)
        return dmudn * self.total_moles