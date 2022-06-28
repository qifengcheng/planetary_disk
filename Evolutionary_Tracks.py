import scipy.interpolate
import astropy.table
import numpy
import time
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString


def read_pms_data(tracks="BHAC15"):
    # Load in the data for the appropriate set of evolutionary tracks.

    path = './data/evolutionary_tracks/'

    if tracks == "BHAC15":
        f = open(path + "BHAC15_tracks+structure.txt", "r")
        lines = f.readlines()
        f.close()

        colnames = lines[46].split()[1:]

        data = numpy.loadtxt(path + "BHAC15_tracks+structure.txt", comments="!", \
                             skiprows=45)
    elif tracks == "Siess2000":
        f = open(path + "siess_2000/m0.13z02.hrd")
        lines = f.readlines()
        f.close()

        line1 = lines[0].replace(" (", "_(").replace("log g", "logg"). \
            replace("#", "").replace("_(Lo)", "/Ls").replace("age_", "log_t"). \
            split()
        line2 = lines[1].replace(" (", "_(").replace("log g", "logg"). \
            replace("#", "").replace("_(Mo)", "/Ms").split()

        colnames = line1 + line2
        colnames[0::2] = line1
        colnames[1::2] = line2

        files = glob.glob(path + "siess_2000/*.hrd")
        for file in files:
            try:
                data = numpy.concatenate((data, numpy.loadtxt(file)))
            except:
                data = numpy.loadtxt(file)

        # Fix the stellar luminosity.

        data[:, 2] = numpy.log10(data[:, 2])
        data[:, -1] = numpy.log10(data[:, -1])
    elif tracks == "Dotter2008":
        f = open(path + "dotter2008/m200fehp00afep0.trk")
        lines = f.readlines()
        f.close()

        # Get the column names.

        colnames = ['M/Ms'] + lines[1].replace("Log ", "Log"). \
                                  replace("Age ", "log_t").replace("LogT", "Teff"). \
                                  replace("LogL", "L/Ls").replace("yrs", "yr").split()[1:]

        # Now read in the data files.

        files = glob.glob(path + "dotter2008/*.trk")
        for file in files:
            new_data = numpy.loadtxt(file)

            # add a column with the mass of the star.

            mass = float(file.split("/")[-1][1:4]) / 100.
            mass_arr = numpy.zeros((new_data.shape[0], 1)) + mass
            new_data = numpy.hstack((mass_arr, new_data))

            # Merge with existing data.

            try:
                data = numpy.concatenate((data, new_data))
            except:
                data = new_data.copy()

        # Get rid of ages more than ~50Myr.

        good = data[:, 1] < 50.0e6
        data = data[good, :]

        # Fix some of the columns.

        data[:, 1] = numpy.log10(data[:, 1])
        data[:, 2] = 10. ** data[:, 2]

    elif tracks == "Tognelli2011":
        f = open(path + "tognelli2011/Z0.02000_Y0.2700_XD2E5_ML1.68_AS05/"
                        "TRK_M0.20_Z0.02000_Y0.2700_XD2E5_ML1.68_AS05.DAT")
        lines = f.readlines()
        f.close()

        # Get the column names.

        colnames = ['M/Ms'] + lines[3].replace("LOG ", "LOG_"). \
                                  replace("LOG_AGE", "log_t(yr)").replace("LOG_L", "L/Ls"). \
                                  replace("LOG_TE", "Teff").split()[1:]

        # Now read in the data files.

        files = glob.glob(path + "tognelli2011/"
                                 "Z0.02000_Y0.2700_XD2E5_ML1.68_AS05/*.DAT")
        for file in files:
            new_data = numpy.loadtxt(file)

            # add a column with the mass of the star.

            mass = float(file.split("/")[-1].split("_")[1][1:])
            mass_arr = numpy.zeros((new_data.shape[0], 1)) + mass
            new_data = numpy.hstack((mass_arr, new_data))

            # Merge with existing data.

            try:
                data = numpy.concatenate((data, new_data))
            except:
                data = new_data.copy()

        # Fix some of the columns.

        data[:, 5] = 10. ** data[:, 5]

    elif tracks == "Feiden2016":
        f = open(path + "feiden2016/std/"
                        "m0090_GS98_p000_p0_y28_mlt1.884.trk")
        lines = f.readlines()
        f.close()

        # Get the column names.

        colnames = ['M/Ms'] + lines[3].replace("Log ", "Log"). \
                                  replace("Age ", "log_t").replace("LogT", "Teff"). \
                                  replace("LogL", "L/Ls").replace("yrs", "yr").split()[1:6]

        # Now read in the data files.

        files = glob.glob(path + "feiden2016/std/*.trk")
        for file in files:
            new_data = numpy.loadtxt(file, usecols=(0, 1, 2, 3, 4))

            # add a column with the mass of the star.

            mass = float(file.split("/")[-1][1:5]) / 1000.
            mass_arr = numpy.zeros((new_data.shape[0], 1)) + mass
            new_data = numpy.hstack((mass_arr, new_data))

            # Merge with existing data.

            try:
                data = numpy.concatenate((data, new_data))
            except:
                data = new_data.copy()

        # Get rid of ages more than ~50Myr.

        good = data[:, 1] < 50.0e6
        data = data[good, :]

        # Fix some of the columns.

        data[:, 1] = numpy.log10(data[:, 1])
        data[:, 2] = 10. ** data[:, 2]

    elif tracks == "Feiden2016mag":
        f = open(path + "feiden2016/mag/"
                        "m1700_GS98_p000_p0_y28_mlt1.884_mag08kG.ntrk")
        lines = f.readlines()
        f.close()

        # Get the column names.

        colnames = ['M/Ms'] + lines[8].replace("conv. ", "conv."). \
                                  replace("AGE", "log_t(yr)").replace("log(Teff)", "Teff"). \
                                  replace("log(L/Lsun)", "L/Ls").replace("Model #", "Model#"). \
                                  replace("M He core", "M_He_core").replace(",", "").split()[1:]

        # Now read in the data files.

        files = glob.glob(path + "feiden2016/mag/*.ntrk")
        for file in files:
            new_data = numpy.loadtxt(file, usecols=tuple([i for i in range(12)]))

            # add a column with the mass of the star.

            mass = float(file.split("/")[-1][1:5]) / 1000.
            mass_arr = numpy.zeros((new_data.shape[0], 1)) + mass
            new_data = numpy.hstack((mass_arr, new_data))

            # Merge with existing data.

            try:
                data = numpy.concatenate((data, new_data))
            except:
                data = new_data.copy()

        # Fix some of the columns.

        data[:, 3] = numpy.log10(data[:, 3] * 1.0e9)
        data[:, 7] = 10. ** data[:, 7]

    elif tracks == "Chen2014":
        f = open(path + "bressan2012/Z0.017Y0.279/"
                        "Z0.017Y0.279OUTA1.77_F7_M000.700.DAT")
        lines = f.readlines()
        f.close()

        # Get the column names.

        colnames = lines[0].replace("LOG ", "LOG_"). \
            replace("AGE", "log_t(yr)").replace("LOG_L", "L/Ls"). \
            replace("LOG_TE", "Teff").replace("MASS", "M/Ms").split()

        # Now read in the data files.

        files = glob.glob(path + "bressan2012/Z0.017Y0.279/*.DAT")

        for file in files:
            new_data = numpy.loadtxt(file, skiprows=2)

            # Merge with existing data.

            try:
                data = numpy.concatenate((data, new_data))
            except:
                data = new_data.copy()

        # Get rid of ages more than ~50Myr.

        good = data[:, 2] < 50.0e6
        data = data[good, :]

        # Fix some of the columns.

        data[:, 2] = numpy.log10(data[:, 2])
        data[:, 4] = 10. ** data[:, 4]

    elif tracks == "Bressan2012":
        f = open(path + "bressan2012/bressan2012.dat")
        lines = f.readlines()
        f.close()

        # Get the column names.

        colnames = lines[13].replace("log(age/yr)", "log_t(yr)"). \
                       replace("logL/Lo", "L/Ls").replace("logTe", "Teff"). \
                       replace("M_act", "M/Ms").split()[1:]

        # Now read in the data files.

        data = numpy.loadtxt(path + "bressan2012/bressan2012.dat")

        # Fix some of the columns.

        data[:, 5] = 10. ** data[:, 5]

    # Make the data into a table.

    table = astropy.table.Table(data, names=colnames)

    # Return the table now.

    return table


def pms_get_mstar(age=None, luminosity=None, tracks="BHAC15"):
    # Load in the data for the appropriate set of evolutionary tracks.

    table = read_pms_data(tracks=tracks)

    # Now do the 2D interpolation.

    Mstar = scipy.interpolate.LinearNDInterpolator((table["L/Ls"], \
                                                    table["log_t(yr)"]), table["M/Ms"], rescale=True)

    # Finally, get the stellar mass.

    if isinstance(age, float) and isinstance(luminosity, float):
        xi = numpy.array([[luminosity, numpy.log10(age)]])
    elif isinstance(age, float):
        xi = numpy.array([[luminosity[i], numpy.log10(age)] for i in range(len(luminosity))])
    else:
        xi = numpy.array([[luminosity[i], numpy.log10(age[i])] for i in \
                          range(len(age))])

    return Mstar(xi)


def pms_get_teff(luminosity=None, age=1.0e6, tracks="BHAC15"):
    # Load in the data for the appropriate set of evolutionary tracks.

    table = read_pms_data(tracks=tracks)

    # Now do the 2D interpolation.

    Teff = scipy.interpolate.LinearNDInterpolator((table["L/Ls"], \
                                                   table["log_t(yr)"]), table["Teff"])

    # Finally, get the stellar mass.

    if isinstance(age, float) and isinstance(luminosity, float):
        xi = numpy.array([[luminosity, numpy.log10(age)]])
    elif isinstance(age, float):
        xi = numpy.array([[luminosity[i], numpy.log10(age)] for i in range(len(mass))])
    else:
        xi = numpy.array([[luminosity[i], numpy.log10(age[i])] for i in range(len(age))])

    return Teff(xi)


def pdspy_get_teff(mass=1.0, age=1.0e6, tracks="BHAC15"):
    # Load in the data for the appropriate set of evolutionary tracks.

    table = read_pms_data(tracks=tracks)

    # Now do the 2D interpolation.

    Teff = scipy.interpolate.LinearNDInterpolator((table["M/Ms"], \
                                                   table["log_t(yr)"]), table["Teff"])

    # Finally, get the stellar mass.

    if isinstance(age, float) and isinstance(mass, float):
        xi = numpy.array([[mass, numpy.log10(age)]])
    elif isinstance(age, float):
        xi = numpy.array([[mass[i], numpy.log10(age)] for i in range(len(mass))])
    else:
        xi = numpy.array([[mass, numpy.log10(age[i])] for i in range(len(age))])

    return Teff(xi)


def pdspy_get_luminosity(mass=1.0, age=1.0e6, tracks="BHAC15"):
    # Load in the data for the appropriate set of evolutionary tracks.

    table = read_pms_data(tracks=tracks)

    # Now do the 2D interpolation.

    Lstar = scipy.interpolate.LinearNDInterpolator((table["M/Ms"], \
                                                    table["log_t(yr)"]), table["L/Ls"])

    # Finally, get the stellar mass.

    if isinstance(age, float) and isinstance(mass, float):
        xi = numpy.array([[mass, numpy.log10(age)]])
    elif isinstance(age, float):
        xi = numpy.array([[mass[i], numpy.log10(age)] for i in range(len(mass))])
    else:
        xi = numpy.array([[mass, numpy.log10(age[i])] for i in range(len(age))])

    return 10. ** Lstar(xi)

def plot_evolutionary_track(track,img_save_direc,img_save_name,if_plot_interpolation_process):
    colors_yr = ["#D4E6F1", "#A9CCE3", "#7FB3D5", "#5499C7", "#2980B9", "#2471A3", "#1F618D", "#1A5276", "#154360"]
    colors_mass = ["#F9E79F", "#F9E79F", "#F7DC6F", "#F7DC6F", "#F4D03F", "#F4D03F", "#F1C40F", "#D4AC0D", "#B7950B",
                   "#9A7D0A", "#7D6608", "#7D6608"]
    # plot evolutionary tracks

    ag = np.arange(0.5, 5, 0.5)
    ms = np.arange(0.2, 1.4, 0.1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1.2]})
    ax = axs[0]
    axs[1].axis("off")

    if if_plot_interpolation_process:
        for mas in ms:
            lum = pdspy_get_luminosity(mass=mas, age=ag * 1e6, tracks="BHAC15")
            tef = pdspy_get_teff(mass=mas, age=ag * 1e6, tracks="BHAC15")
            ax.plot(tef, lum, label=f'{mas:.1f} M_sun', c=colors_mass[np.where(ms == mas)[0][0]])

            if mas == ms[3]:
                f = lum
                x1 = tef

        for agg in ag:
            lum = pdspy_get_luminosity(mass=ms, age=agg * 1e6, tracks="BHAC15")
            tef = pdspy_get_teff(mass=ms, age=agg * 1e6, tracks="BHAC15")
            ax.plot(tef, lum, label=f'{agg} yr', c=colors_yr[np.where(ag == agg)[0][0]])

            if agg == 1.0:
                g = lum
                x2 = tef

        line_1 = LineString(np.column_stack((x1, f)))
        line_2 = LineString(np.column_stack((x2, g)))
        intersection = line_1.intersection(line_2)

        ax.plot(x1, f, 'c', linestyle='dashed', label='Provided Age')
        ax.axhline(y=intersection.y, linestyle='dashed', color='c', label='Provided Luminosity')
        ax.plot(*intersection.xy, 'ro', label='intersection point')
        ax.plot(x2, g, 'r', linestyle='dashed', label='Derived Mass')


    else:
        for mas in ms:
            lum = pdspy_get_luminosity(mass=mas, age=ag * 1e6, tracks=track)
            tef = pdspy_get_teff(mass=mas, age=ag * 1e6, tracks=track)
            ax.plot(tef, lum, label=f'{mas:.1f} M_sun', c=colors_mass[np.where(ms == mas)[0][0]])

        for agg in ag:
            lum = pdspy_get_luminosity(mass=ms, age=agg * 1e6, tracks=track)
            tef = pdspy_get_teff(mass=ms, age=agg * 1e6, tracks=track)
            ax.plot(tef, lum, label=f'{agg} yr', c=colors_yr[np.where(ag == agg)[0][0]])

    ax.set_ylabel('$L_{bol}$/$L_{sun}$', fontsize=16)
    ax.set_xlabel('Effective temperature/K', fontsize=16)
    ax.set_title(f'Interpolation of Evolutionary Tracks {track}')
    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    direc = img_save_direc
    os.makedirs(direc, exist_ok=True)
    plt.savefig(os.path.join(direc, f'{img_save_name}.pdf'))
    plt.close()

def main():
    img_save_name = 'Evolutionary_tracks'
    direc = 'pictures_evolutionary_tracks'
    track = "BHAC15"

    plot_evolutionary_track(track,direc,img_save_name,False)

    img_save_name_2 = 'Evolutionary_tracks_with_interpolation_proc'
    plot_evolutionary_track(track, direc, img_save_name_2, True)

if __name__ == '__main__':
    main()


