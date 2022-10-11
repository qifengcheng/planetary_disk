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
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Evolutionary Tracks', fontsize=16)


def plot_violin(mass_1y_e056, mass_1y_en01, mass_05y_e056, mass_05y_en01, name):
    fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15,10), sharey=True)
    ax1.set_title(f'{name}'+ r", $L_{bol} \propto \nu^{0.56}$" + r", 1 $Myr$", fontsize=20)
    ax1.set_ylabel('$M_*$ (M$_{\odot}$)', fontsize=16)
    ax1.violinplot(mass_1y_e056)
    ax2.set_title(f'{name}'+ r", $L_{bol} \propto \nu^{-0.1}$" + r", 1 $Myr$", fontsize=20)
    ax2.set_ylabel('$M_*$ (M$_{\odot}$)', fontsize=16)
    ax2.violinplot(mass_1y_en01)
    ax3.set_title(f'{name}'+ r", $L_{bol} \propto \nu^{0.56}$" + r", 0.5 $Myr$", fontsize=20)
    ax3.set_ylabel('$M_*$ (M$_{\odot}$)', fontsize=16)
    ax3.violinplot(mass_05y_e056)
    ax4.set_title(f'{name}'+ r", $L_{bol} \propto \nu^{-0.1}$" + r", 0.5 $Myr$", fontsize=20)
    ax4.set_ylabel('$M_*$ (M$_{\odot}$)', fontsize=16)
    ax4.violinplot(mass_05y_en01)

    labels = ["BHAC15", "Siess2000", "Dotter2008", "Tognelli2011", "Feiden2016", "Feiden2016mag", "Chen2014",
              "Bressan2012"]

    for ax in [ax1, ax2, ax3, ax4]:
        set_axis_style(ax, labels)
    fig.tight_layout()
    direc = 'pictures_violin'
    os.makedirs(direc, exist_ok=True)
    plt.savefig(os.path.join(direc, f'{name}_violin_plot.pdf'))
    plt.close()


# plot all sources under the same track and conditions
def plot_violin_same_track(mass_all_sources, trackname, condition, labels, color=False):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.violinplot(mass_all_sources)
    axs.set_ylabel('$M_*$ (M$_{\odot}$)', fontsize=16)

    if condition == 'prop056_1myr':
        plt.title(f'{trackname}' + r", $L_{bol} \propto \nu^{0.56}$" + r", 1 $Myr$", fontsize=20)
    elif condition == 'prop056_05myr':
        plt.title(f'{trackname}' + r", $L_{bol} \propto \nu^{0.56}$" + r", 0.5 $Myr$", fontsize=20)
    elif condition == 'propn1_1myr':
        plt.title(f'{trackname}' + r", $L_{bol} \propto \nu^{-0.1}$" + r", 1 $Myr$", fontsize=20)
    elif condition == 'propn1_05myr':
        plt.title(f'{trackname}' + r", $L_{bol} \propto \nu^{-0.1}$" + r", 0.5 $Myr$", fontsize=20)
    else:
        raise Exception("Illegal filename")
    set_axis_style(axs, labels)
    axs.set_xlabel('Sources', fontsize=16)
    direc = 'pictures_violin'
    os.makedirs(direc, exist_ok=True)
    plt.savefig(os.path.join(direc, f'{trackname}_{condition}_violin_plot.pdf'), bbox_inches='tight')
    plt.close()


def read_plot_violin_same_track(objs, trackname='BHAC15'):
    # read mass data from previous calculation
    direc = './data_result'
    conditions = ['prop056_1myr', 'prop056_05myr', 'propn1_1myr', 'propn1_05myr']

    for condition in conditions:
        mass_all_sources_same_track = []
        labels = []
        for obj in objs:
            df = pd.read_csv(os.path.join(direc, f'{obj["name"]}_mass_tabel_{condition}.csv'))
            mass_all_sources_same_track.append(df[trackname])
            labels.append(obj['name'])
        plot_violin_same_track(mass_all_sources_same_track, trackname, condition, labels)


def save_mass_distribution_tabel(data, file_name, direc):
    trac = ["BHAC15", "Siess2000", "Dotter2008", "Tognelli2011", "Feiden2016", "Feiden2016mag", "Chen2014",
            "Bressan2012"]

    mass_tabel = pd.DataFrame({trac[i]: data[i] for i in range(8)})
    os.makedirs(direc, exist_ok=True)
    mass_tabel.to_csv(os.path.join(direc, f'{file_name}.csv'))
    return mass_tabel

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


    objs = [obj_56, obj_56_3RMS, obj_65, obj_65_3RMS, obj_124, obj_124_3RMS, obj_56_3RMS, obj_140, obj_140_3RMS, obj_157_a, obj_157_b, obj_157_3RMS, obj_163_3RMS, obj_270_3RMS]
    tracks = ["BHAC15", "Siess2000", "Dotter2008", "Tognelli2011", "Feiden2016", "Feiden2016mag", "Chen2014",
              "Bressan2012"]

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

        # create pandas dataframe for masses and save to csv files
        save_mass_distribution_tabel(mass_al_trac_1y, f'{obj["name"]}_mass_tabel_prop056_1myr', 'data_result')
        save_mass_distribution_tabel(mass_al_trac_1yn, f'{obj["name"]}_mass_tabel_propn1_1myr', 'data_result')
        save_mass_distribution_tabel(mass_al_trac_05y, f'{obj["name"]}_mass_tabel_prop056_05myr', 'data_result')
        save_mass_distribution_tabel(mass_al_trac_05yn, f'{obj["name"]}_mass_tabel_propn1_05myr', 'data_result')

    # read mass distribution and plot mass distribution of all sources under the same track and conditions
    for track in tracks:
        read_plot_violin_same_track(objs, track)

if __name__ == '__main__':
    main()
