import numpy as np
import pandas as pd
import os
import inspect
import sys
import sncosmo


def add_ztf_filts(ztf_g = True, ztf_r = True):
    """Add ZTF filter curves to the sncosmo registry
    
    Parameters
    ----------
    ztf_g : boolean (default=True)
        Add the ZTF g-band filter to sncosmo
    
    ztf_r : boolean (default=True)
        Add the ZTF r-band filter to sncosmo
    """
    
    module_path = os.path.abspath(inspect.getsourcefile(add_ztf_filts))[:-19]
    
    wave_g, trans_g = np.loadtxt(module_path+"filters/ztf_g.csv", 
                                 delimiter=",", unpack=True, usecols=(0,1))
    wave_r, trans_r = np.loadtxt(module_path+"filters/ztf_r.csv", 
                                 delimiter=",", unpack=True, usecols=(0,1))

    wavelength = 10*np.linspace(min(wave_g), max(wave_g), len(wave_g))
    
    filters = {}
    filters["ztf_g"] = np.array( [(w, t) for w, t in 
                                 zip(wavelength, trans_g/100)], 
                                 dtype=[("wavelength", "d"), 
                                 ("transmission", "d")])
    filters["ztf_r"] = np.array( [(w, t) for w, t in 
                                 zip(wave_r*10, trans_r/100)], 
                                 dtype=[("wavelength", "d"), 
                                 ("transmission", "d")])
    for key in filters:
        band = sncosmo.Bandpass(filters[key]["wavelength"],
                                filters[key]["transmission"],
                                name=key)
        sncosmo.registry.register(band)
    
class IaLightCurve:
    
    def __init__(self, z):
        """Constructor for the light curve class"""
        self.z = z
        return
    
    def define_sncosmo_model(self, t0 = 0.0, 
                             hostebv = 0.0, mwebv = 0.0,
                             x1 = 0.0, c = 0.0, 
                             abs_mag_b_peak = -19.35):
        """Define the model to produce model light curve
        
        Parameters
        ----------
        z : float
            The SN redshift for output light curve
        
        t0 : float, optional, default: 0.0
            The time of zero phase for the SN
            
            NOTE this is not B-band maximum in sncosmo
        
        hostebv : float, optional, default: 0.0
            Host-galaxy redenning, E(B - V)
        
        mwebv : float, optional, default: 0.0
            Milky Way redenning, E(B - V)
        
        x1 : float, optional, default: 0.0
            SALT-II x1 parameter
                             
        c : float, optional, default: 0.0
            SALT-II c parameter

        abs_mag_b_peak : float, optional, default: -19.35
            Absolute mag in B band at the time of peak
            (This parameter is needed to normalize )          
                
        Examples
        --------
        >>> from ztf_ia_lcs.ztf_gr_lc import IaLightCurve
        >>> ztf_ia_z0_1 = IaLightCurve(0.1).define_sncosmo_model()
        """
        
        dust = sncosmo.F99Dust()
        host_dust = sncosmo.F99Dust()
        model = sncosmo.Model(source="salt2",
                              effects=[host_dust, dust],
                              effect_names=["host", "mw"],
                              effect_frames=["rest", "obs"])
        model.set(z = self.z)
        model.set(t0 = t0)
        model.set(hostebv = hostebv)
        model.set(mwebv = mwebv)
        model.set(x1 = x1)
        model.set(c = c)
        model.set_source_peakabsmag(abs_mag_b_peak, 'bessellb', 'ab')
        
        self.model = model
        
    def generate_lc(self, filters, t_grid = np.arange(-15,40,0.1)):
        """Generate SALT-II light curves in the specified filters
        
        Parameters
        ----------
        filters : list
            List of filters for output light curve
        
        t_grid : array-like, optional, (default=np.arange(-15,40,0.1))
            Array-like grid of times over which the light curve 
            is to be evaluated.
        
        Attributes
        ----------
        t_grid_ : array-like
            Array with observed 
        
        lcs_ : dict
            Light curves associated with each filter evalutated at t_grid
            form ``{filter: light_curve}''
        
        Examples
        --------
        >>> from ztf_ia_lcs.ztf_gr_lc import IaLightCurve
        >>> ztf_ia_z0_1 = IaLightCurve(0.1).define_sncosmo_model()
        >>> ztf_ia_z0_1.generate_lc(["ztf_g", "ztf_r"])
        """
        try:
            self.t_grid_ = t_grid*(1 + self.z)
            self.lcs_ = {}
            for filt in filters:
                self.lcs_[filt] = self.model.bandmag(filt, "ab", t_grid)
        except AttributeError:
            print("""sncosmo model not defined. Try define_sncosmo_model()""")
    
    def create_lc_file(self):
        """Print all simulated light curves to csv file"""
        
        try:
            lc_df = pd.DataFrame(self.t_grid_, columns = ["time"])
            for key, value in self.lcs_.items():
                lc_df[key] = value
            lc_df.to_csv("ztf_ia_z{:.2f}.csv".format(self.z), index = False)
        except AttributeError:
            print("""No LCs associated with LC object. Try generate_lc()""")

def main(argv):
    add_ztf_filts()
    lc = IaLightCurve(argv)
    lc.define_sncosmo_model()
    lc.generate_lc(filters=["ztf_g","ztf_r"])
    lc.create_lc_file()

if __name__ == "__main__":
    main(float(sys.argv[1]))
    