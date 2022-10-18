__author__=='sebastiangrandis', 'giovanniarico'

import numpy as np
import baccoemu


def virial_mass_converter(delta1, mass1, conc1, delta2, conc2):
    """
    functions to compute halo mass
    considering an overdensity delta1 starting from a different overdensity delta2
    and assuming a NFW halo profile
    e.g. how to go from Delta_200 to Delta_500
    see e.g. Ettori et al 2011
    """
    mass2 = delta2*conc2**3/(delta1*conc1**3)*mass1
    return mass2


def virial_concentration_converter(delta1, conc1, delta2):
    """
    functions to compute halo concentration
    changing value of overdensity and assuming a NFW halo profile
    e.g. how to go from Delta_200 to Delta_500
    see e.g. Ettori et al 2011
    """
    from scipy import optimize

    def function(conc2):
        return (conc1/conc2)**3 * (np.log(1+conc2)-conc2/(1+conc2)) / (
            np.log(1+conc1)-conc1/(1+conc1)) - delta2/delta1
    conc2 = optimize.broyden1(function, conc1)
    return conc2


def gas_fraction_converter(mass200, delta500, mass500, params, delta200=200):
    """
    function to transform the BG mass fraction within r_delta1 w.r.t. mass_delta1, 
    into the BG mass fraction within r_delta2 w.r.t. mass_delta2. 
    Note that the parameters of the baryonification model need to be defined w.r.t. delta1=200c
    """
    
    # compute the ratio between M_BG(<r_delta2) / M_BG(<r_delta1)
    u = np.logspace(-3, 0, num=200)
    beta = 3 - (10**params['M_inn']/mass200)**(10**params['eta'])
    integrand = u**2 / ( 1 + u/10**params['theta_inn'] )**beta / ( 1 + (u/10**params['theta_out'])**2  )**2
    u_500 = (mass500/mass200*delta200/delta500)**(1/3)
    
    mask = np.where(u<u_500, 1, 0)
    numer = np.trapz(integrand*mask, u)
    denom = np.trapz(integrand, u)

    return numer/denom*mass200/mass500

    
def predictor(M500c, params, compo, eps=0.05):
    """
    function compute the compo='bo_gas', 'stellar' mass fraction and slope at the pivot mass M500c [Msol, no little h] 
    for the cosmological and BCM parameter params
    """
    
    mm = np.array([M500c*np.exp(-eps), M500c, M500c*np.exp(eps)])
    
    # Ragagnin+21 relation at pivot cosmology
    c500c = np.exp( 0.86 - 0.05*np.log(mm/13.7e13) )
    
    c200c = np.array([virial_concentration_converter(500, c, 200) for c in c500c])
    M200c = virial_mass_converter(500, mm, c500c, 200, c200c)
    
    if compo=='bo_gas':
        corr = np.array([gas_fraction_converter(m200, 500, m500, params) for m200, m500 in zip(M200c, mm)])
    elif compo=='stellar':
        corr = 1
    
    # bacco takes masses in units of Msol/h, while we worked in Msol
    fracs = baccoemu.get_baryon_fractions(M200c/params['hubble'], **params)[compo]*corr
    
    slope = 0.5*np.log(fracs[2]/fracs[0])/eps
    
    return fracs[1], slope

# data presented in Grandis+22
data_WtG = {'WtG': {'Mpiv': 1e15, 'type':'bo_gas', 'frac': np.exp(-2.08), 'dfrac': 0.04*np.exp(-2.08), 
                'slope':0.001, 'dslope':10.}, # the error on the slope here is very large, s.t. it doesn't impact the fit
           }

data_SPT = {'SPT_ICM': {'Mpiv': 4.8e14, 'type':'bo_gas', 'frac': 5.69e13/4.8e14, 'dfrac': 0.62/5.69*5.69e13/4.8e14, 
                    'slope':0.33, 'dslope':0.09},
            'SPT_stars': {'Mpiv': 4.8e14, 'type':'stellar', 'frac': 4e12/4.8e14, 'dfrac': 0.28/4*4e12/4.8e14, 
                    'slope':-0.2, 'dslope':0.12},
           }

data_eFEDS = { 'eFEDS': {'Mpiv': 2e14, 'type':'bo_gas', 'frac': 1.08e13/2e14, 'dfrac': 0.13/1.08*1.08e13/2e14, 
                    'slope':0.19, 'dslope':0.5*(0.099+0.118)},  
             }

data_XXL = { 'HSC-XXL': {'Mpiv': 1e14, 'type':'bo_gas', 'frac': np.exp(2.02)*1e-2, 'dfrac': 0.1*np.exp(2.02)*1e-2, 
                    'slope':0.23, 'dslope':0.12, } ,
            'HSC-XXL_stars': {'Mpiv': 1e14, 'type':'stellar', 'frac': np.exp(0.75)*1e-2, 'dfrac': 0.13*np.exp(0.75)*1e-2, 
                    'slope':-0.2, 'dslope':0.11}
             }

# mapping when constraining the BCM params, as used in Grandis+22
mapping = {'M_c': 0, 'beta':1, 'M1_z0_cen':2}
        
const = { 'omega_cold'    :  0.315,
         'sigma8_cold'   :  0.83,
         'omega_baryon'  :  0.05,
        'ns'            :  0.96,
        'hubble'        :  0.67,
        'neutrino_mass' :  0.0,
        'w0'            : -1.0,
        'wa'            :  0.0,
        'expfactor'     :  1,
    
        'M_c'           :  14.5,
        'eta'           : -0.248,
        'beta'          : -0.22,
        'M1_z0_cen'     : 10.5,
        'theta_out'     : 0.419,
        'theta_inn'     : -0.702,
        'M_inn'         : 13.0
}

class ClusterDataLikelihood(object):
    """
    Likelihood module to constrain the Arico-BCM with cluster and group data
    """
    
    def __init__(self, mapping, data, constants=const, only_fracs=False):
        """
        mapping: dict, keys are param names, while values are indices of samling vector, where that param is stored
        data: dict, contains the cluster and group data, see above for examples
        constants: dict, keys are param names, values are values of the params, follows the bacco convention
        """
        
        self.data = data
        
        self.only_fracs = only_fracs
        
        self.constants = constants
        self.mapping = mapping
        
    def setup(self):
        pass
        
    def computeLikelihood(self, ctx):
        """
        likelihood calling in cosmoHammer, ignore when using other sampler
        """
        
        p1 = np.array(ctx.getParams())
        
        return self.getLike(p1)
    
    def _vec2params(self, p1):
        """
        transforms sampling vector into params dictionary for later computations
        """
        
        params = self.constants.copy()
        
        for k, v in self.mapping.items():
            params[k] = p1[v]
            
        return params
    
    def getLike(self, p1):
        """
        compute the likelihood as function of the sampling parameters p1
        """
        
        # get the all the params needed for bacco
        params = self._vec2params(p1)
        
        # compute the fraction and slope predictions
        if self.only_fracs:
            preds = { kk: {'frac': predictor(vv['Mpiv'], params, vv['type'])[0],} for kk, vv in self.data.items()}
        else:
            preds = { kk: {'frac': predictor(vv['Mpiv'], params, vv['type'])[0],
               'slope': predictor(vv['Mpiv'], params, vv['type'])[1]} for kk, vv in self.data.items()}
        
        # sum up the likelihood of each data point
        lnL = 0 
        for k, v in preds.items():
            for kk, vv in v.items():
                lnL += -0.5*(self.data[k][kk]-vv)**2 /  self.data[k]['d'+kk]**2
                
        return lnL
        