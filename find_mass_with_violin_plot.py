import scipy.interpolate
import astropy.table
import numpy
import time
import glob
import numpy as np
import numpy.random as random
import statistics
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from find_bol_L import monte_carlo, find_flux
from evol_track_interpolation import pms_get_mstar


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

    
def plot_violin(mass_1y_e056, mass_1y_en01, mass_05y_e056, mass_05y_en01, name):
    fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15,10), sharey=True)
    ax1.set_title(f'{name}_0.56exp_1byr')
    ax1.set_ylabel('mass/M_sun')
    ax1.violinplot(mass_1y_e056)
    ax2.set_title(f'{name}_n0.1exp_1byr')
    ax2.set_ylabel('mass/M_sun')
    ax2.violinplot(mass_1y_en01)
    ax3.set_title(f'{name}_0.56exp_0.5byr')
    ax3.set_ylabel('mass/M_sun')
    ax3.violinplot(mass_05y_e056)
    ax4.set_title(f'{name}_n0.1exp_0.5byr')
    ax4.set_ylabel('mass/M_sun')
    ax4.violinplot(mass_05y_en01)

    labels = ["A", "B", "C","D","E","F","G","H"]
    for ax in [ax1, ax2, ax3, ax4]:
        set_axis_style(ax, labels)
    fig.tight_layout()
    direc = 'pictures_violin'
    os.makedirs(direc, exist_ok=True)
    plt.savefig(os.path.join(direc, f'{name}.png'))
    plt.close()

def main():

    obj_56 = {'name': 'HOPS-56', 'frequency': 33.0e9, 'flux':5.69e-5}
    obj_65 = {'name': 'HOPS-65', 'frequency': 15.0e9, 'flux':1.064e-5}
    obj_124 = {'name': 'HOPS-124', 'frequency': 44.0e9, 'flux': 0.0005424}
    obj_140 = {'name': 'HOPS-140', 'frequency': 33.0e9, 'flux':0.0001537}
    obj_157_a = {'name': 'HOPS-157_a', 'frequency': 33.0e9, 'flux':2.67e-5}
    obj_157_b = {'name': 'HOPS-157_b', 'frequency': 33.0e9, 'flux':2.34e-5}
    obj_163_3RMS = {'name': 'HOPS-163', 'frequency': 33.0e9, 'flux':4.87e-6*3}
    obj_270_3RMS = {'name': 'HH270MMS2', 'frequency': 44.0e9, 'flux':8.75e-6*3}

    obj_56_3RMS = {'name': 'HOPS-56', 'frequency': 33.0e9, 'flux':7.81e-6*3}
    obj_65_3RMS = {'name': 'HOPS-65', 'frequency': 15.0e9, 'flux':2.44e-6*3}
    obj_124_3RMS = {'name': 'HOPS-124', 'frequency': 44.0e9, 'flux': 6.54e-5*3}
    obj_140_3RMS = {'name': 'HOPS-140', 'frequency': 33.0e9, 'flux':7.54e-6*3}
    obj_157_3RMS = {'name': 'HOPS-157', 'frequency': 33.0e9, 'flux':9.49e-6*3}


    objs = [obj_56, obj_56_3RMS, obj_65, obj_65_3RMS, obj_124, obj_56_3RMS, obj_140, obj_140_3RMS, obj_157_a, obj_157_b, obj_157_3RMS, obj_163_3RMS, obj_270_3RMS]


    for obj in objs:
        print(obj)
        lam_in = 3.0e8/(obj['frequency'])
        lam_out = 4.1e-2
        fluxin = obj['flux']
        flux_41_01 = find_flux(fluxin*1000, lam_in, lam_out, -0.1)
        flux_41_051 = find_flux(fluxin*1000, lam_in, lam_out, 0.51)

        trac = ["BHAC15","Siess2000",  "Dotter2008", "Tognelli2011","Feiden2016","Feiden2016mag","Chen2014","Bressan2012"]

        L_bol_056, L_mean_056, L_std_056 = monte_carlo(flux=flux_41_051, D=0.4)
        L_bol_n01, L_mean_n01, L_std_n01 = monte_carlo(flux=flux_41_01, D=0.4)

        mass_al_trac_1y = np.empty(len(trac), dtype=object)
        mass_al_trac_1yn = np.empty(len(trac), dtype=object)
        mass_al_trac_05y = np.empty(len(trac), dtype=object)
        mass_al_trac_05yn = np.empty(len(trac), dtype=object)

        for track in trac:
            mass_al_trac_1y[trac.index(track)] = pms_get_mstar(age = 1.0e6, luminosity= L_bol_056, tracks=track)
            mass_al_trac_1yn[trac.index(track)] = pms_get_mstar(age = 1.0e6, luminosity= L_bol_n01, tracks=track)
            mass_al_trac_05y[trac.index(track)] = pms_get_mstar(age = 0.5e6, luminosity=L_bol_056, tracks=track)
            mass_al_trac_05yn[trac.index(track)] = pms_get_mstar(age = 0.5e6, luminosity= L_bol_n01, tracks=track)

        plot_violin(mass_al_trac_1y, mass_al_trac_1yn, mass_al_trac_05y, mass_al_trac_05yn, obj['name'])

if __name__ == '__main__':
    main()
